#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch
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

DIRECT_TMPL = "Question: {q}\nAnswer briefly in one sentence.\nAnswer:"

def final_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def load_probe(npz_path: Path):
    z = np.load(npz_path)
    w = torch.tensor(z["w"], dtype=torch.float32)   # as-trained (no normalization)
    b = float(z["b"])
    d = int(z["d_model"])
    return w, b, d

@torch.no_grad()
def span_mean_h(model, tok, prompt, final, layer_idx, device):
    enc_p = tok(prompt, return_tensors="pt").to(device)
    enc_f = tok(prompt + " " + final, return_tensors="pt").to(device)
    out = model(**enc_f, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[layer_idx]                 # [1, seq, d]
    start = enc_p["input_ids"].shape[1]
    vec = hs[0, start:, :].mean(dim=0)                # [d]
    return vec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=9, help="1-indexed GPT-2 block (your probe layer)")
    ap.add_argument("--probe_file", default="saved_probes/probe_layer9.npz")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4])
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--max_new", type=int, default=60)
    ap.add_argument("--out", default="steer_results.jsonl")
    args = ap.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    # load probe
    w, b, d_model = load_probe(Path(args.probe_file))
    assert model.config.n_embd == d_model, f"Model d_model={model.config.n_embd}, probe d_model={d_model}"

    results = []
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    for q in QS:
        prompt = DIRECT_TMPL.format(q=q)
        enc_prompt = tok(prompt, return_tensors="pt").to(device)
        prompt_len = enc_prompt["input_ids"].shape[1]

        for alpha in args.alphas:
            delta = (alpha * w).view(1, 1, -1)

            def hook_fn(module, inp, out):
                # out may be Tensor or tuple (hidden_states, present, ...)
                if isinstance(out, tuple):
                    hs, *rest = out
                else:
                    hs, rest = out, None
                # steer ALL positions after the prompt
                if hs.shape[1] > prompt_len:
                    hs = hs.clone()
                    hs[:, prompt_len:, :] = hs[:, prompt_len:, :] + delta.to(hs.device)
                return (hs, *rest) if rest is not None else hs

            block = model.transformer.h[args.layer - 1]  # 0-indexed blocks
            handle = block.register_forward_hook(hook_fn)

            with torch.no_grad():
                gen = model.generate(
                    **enc_prompt,
                    max_new_tokens=args.max_new,
                    do_sample=False,          # deterministic for clearer effect
                    num_beams=4,
                    length_penalty=0.0,
                    no_repeat_ngram_size=3,
                    pad_token_id=tok.eos_token_id if tok.pad_token is None else tok.pad_token_id
                )
            handle.remove()

            text = tok.decode(gen[0], skip_special_tokens=True)
            fin = final_line(text)

            # score with same probe at same layer (mean over final sentence tokens)
            h = span_mean_h(model, tok, prompt, fin, args.layer, device)  # [d]
            logit = (h @ w.to(h.device)) + b
            p = torch.sigmoid(logit).item()

            rec = {
                "question": q,
                "alpha": alpha,
                "final_line": fin,
                "probe_logit": float(logit.item()),
                "probe_p_cot": float(p),
                "layer": args.layer
            }
            results.append(rec)
            print(f"[{q[:32]}...] α={alpha:+.2f}  p={p:.3f}  logit={logit.item():+.3f} | {fin[:80]}")

    with open(outp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} rows to {outp.resolve()}")

    # quick alpha summary (mean p and mean logit)
    from collections import defaultdict
    by_alpha = defaultdict(list)
    by_alpha_logit = defaultdict(list)
    for r in results:
        by_alpha[r["alpha"]].append(r["probe_p_cot"])
        by_alpha_logit[r["alpha"]].append(r["probe_logit"])
    summary = {a: {"mean_p": float(np.mean(v)), "std_p": float(np.std(v)), "n": len(v)} for a, v in sorted(by_alpha.items())}
    summary_log = {a: {"mean_logit": float(np.mean(v)), "std_logit": float(np.std(v)), "n": len(v)} for a, v in sorted(by_alpha_logit.items())}
    print("Summary p(CoT) by α:\n", json.dumps(summary, indent=2))
    print("Summary logit by α:\n", json.dumps(summary_log, indent=2))

if __name__ == "__main__":
    main()
