#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QS = [
    "What is the capital of France?",
    "What is 12 + 35?",
    "Who wrote the play Hamlet?",
    "What is the tallest mountain in the world?",
    "Why do we see phases of the Moon?",
    "How many days are in a leap year?",
    "Which planet is known as the Red Planet?",
    "What gas do plants absorb during photosynthesis?",
    "Who discovered penicillin?",
    "What is the boiling point of water at sea level in Celsius?"
]

DIRECT_TMPL = (
    "Question: {q}\n"
    "Answer in EXACTLY five words. No extra words. End with a period.\n"
    "Answer:"
)
COT_TMPL = (
    "Question: {q}\n"
    "Think step by step in 1–2 short sentences.\n"
    "Then write exactly one line: Final answer: <EXACTLY five words ending with a period>\n"
    "Answer: Let's think step by step."
)

# ---------- helpers ----------
SENT_END = re.compile(r"([^.?!]*[.?!])")  # first sentence
def first_sentence(s: str) -> str:
    m = SENT_END.search(s.strip())
    return m.group(1).strip() if m else s.strip()

def clean_line(s: str) -> str:
    return " ".join(s.strip().split())

def word_count_ok(s: str, target=5, tol=0):
    s = s.rstrip(".").strip()
    n = len([w for w in s.split() if w])
    return (target - tol) <= n <= (target + tol)

def extract_final_direct(text: str) -> str:
    # Take text after the first "Answer:" and then the first sentence of that
    after = text.split("Answer:", 1)[-1]
    return clean_line(first_sentence(after))

def extract_final_cot(text: str) -> str:
    # Prefer explicit "Final answer:" line if present
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.lower().startswith("final answer:"):
            cand = ln.split(":", 1)[-1].strip()
            return clean_line(first_sentence(cand))
    # Fallback: last non-empty line’s first sentence
    return clean_line(first_sentence(lines[-1] if lines else ""))

@torch.no_grad()
def gen(model, tok, prompt, device,
        max_new_tokens=40, temperature=0.6, top_p=0.9,
        repetition_penalty=1.12, no_repeat_ngram_size=4):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.eos_token_id if tok.pad_token is None else tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="cot_outputs_paraphrase.jsonl")
    ap.add_argument("--per-q", type=int, default=2)
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--retries", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    saved = 0; failures = 0

    with open(outp, "w", encoding="utf-8") as f:
        for q in QS:
            dp = DIRECT_TMPL.format(q=q)
            cp = COT_TMPL.format(q=q)
            for _ in range(args.per_q):
                # DIRECT
                d_final = ""
                for _ in range(args.retries):
                    d_txt = gen(model, tok, dp, device)
                    d_final = extract_final_direct(d_txt)
                    if word_count_ok(d_final, target=5, tol=0):
                        break
                    d_final = ""
                if not d_final:
                    failures += 1
                    continue

                # CoT
                c_final = ""
                for _ in range(args.retries):
                    c_txt = gen(model, tok, cp, device)
                    c_final = extract_final_cot(c_txt)
                    if word_count_ok(c_final, target=5, tol=0):
                        break
                    c_final = ""
                if not c_final:
                    failures += 1
                    continue

                f.write(json.dumps({"question": q, "mode": "direct", "label": 0,
                                    "prompt": dp, "answer": d_txt, "final_line": d_final})+"\n")
                f.write(json.dumps({"question": q, "mode": "cot", "label": 1,
                                    "prompt": cp, "answer": c_txt, "final_line": c_final})+"\n")
                saved += 2
                print(f"[{saved:03d}] {q} -> finals | D: '{d_final}' | C: '{c_final}'")

    print(f"\nSaved {saved} examples to {outp.resolve()} (failures: {failures})")

if __name__ == "__main__":
    main()
