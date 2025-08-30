#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

QS = [
    "What is the capital of France?",
    "What is 12 + 35?",
    "Who wrote the play Hamlet?",
    "What is the tallest mountain in the world?",
    "Why do we see phases of the Moon?",
    "Is it possible to have negative absolute temperature?",
    "If a train goes 60 mph for 2 hours, how far does it travel?",
    "Who discovered penicillin?",
    "Why is the sky blue?",
    "How many days are in a leap year?",
    "Which planet is known as the Red Planet?",
    "What gas do plants absorb during photosynthesis?"
]

DIRECT_TMPL = "Question: {q}\nAnswer briefly in one sentence.\nAnswer:"
COT_TMPL    = "Question: {q}\nThink step by step, then give a final one-sentence answer.\nAnswer: Let's think step by step."

@torch.no_grad()
def gen_causal(model, tok, prompt, device, max_new_tokens=96, temperature=0.7, top_p=0.9,
               repetition_penalty=1.05, no_repeat_ngram_size=3):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.eos_token_id if tok.pad_token is None else tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

@torch.no_grad()
def gen_seq2seq(model, tok, prompt, device, max_new_tokens=96, num_beams=4):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="causal (gpt2/distilgpt2) or seq2seq (google/flan-t5-small/base)")
    ap.add_argument("--out", default="cot_outputs.jsonl")
    ap.add_argument("--per-q", type=int, default=3, help="pairs per question")
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--seq2seq", action="store_true", help="force seq2seq API")
    args = ap.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    # auto-detect causal vs seq2seq unless overridden
    is_seq2seq = args.seq2seq
    tok = AutoTokenizer.from_pretrained(args.model)
    model = None
    if not is_seq2seq:
        try:
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
            is_seq2seq = False
        except Exception:
            is_seq2seq = True
    if is_seq2seq and model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    with open(outp, "w", encoding="utf-8") as f:
        for q in QS:
            direct_prompt = DIRECT_TMPL.format(q=q)
            cot_prompt    = COT_TMPL.format(q=q)
            for _ in range(args.per_q):
                if is_seq2seq:
                    direct_text = gen_seq2seq(model, tok, direct_prompt, device)
                    cot_text    = gen_seq2seq(model, tok, cot_prompt, device)
                else:
                    direct_text = gen_causal(model, tok, direct_prompt, device)
                    cot_text    = gen_causal(model, tok, cot_prompt, device)

                # store both; label: 0=Direct, 1=CoT
                rec_direct = {
                    "question": q,
                    "mode": "direct",
                    "label": 0,
                    "prompt": direct_prompt,
                    "answer": direct_text.strip(),
                    "full_text": direct_text
                }
                rec_cot = {
                    "question": q,
                    "mode": "cot",
                    "label": 1,
                    "prompt": cot_prompt,
                    "answer": cot_text.strip(),
                    "full_text": cot_text
                }
                f.write(json.dumps(rec_direct, ensure_ascii=False) + "\n")
                f.write(json.dumps(rec_cot, ensure_ascii=False) + "\n")
                saved += 2
                print(f"[{saved:03d}] {q} -> saved Direct & CoT")

    print(f"\nSaved {saved} examples to {outp.resolve()}")

if __name__ == "__main__":
    main()
