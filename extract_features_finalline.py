#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

@torch.no_grad()
def span_mean_vecs(model, tok, prompt, final_line, device, layers):
    # Tokenize prompt then prompt + final line; average activations over final line span
    enc_prompt = tok(prompt, return_tensors="pt").to(device)
    # Add a space to avoid token-merge with last prompt token
    enc_full   = tok(prompt + " " + final_line, return_tensors="pt").to(device)
    out = model(**enc_full, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # [emb, L1, ..., LN]
    start = enc_prompt["input_ids"].shape[1]
    end   = enc_full["input_ids"].shape[1]
    feats = []
    for li in layers:
        H = hs[li][0, start:end, :]  # [answer_len, d_model]
        feats.append(H.mean(dim=0).cpu().numpy())
    return np.concatenate(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cot_outputs.jsonl")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", type=int, nargs="+", default=[3,6,9,12])
    ap.add_argument("--out", default="cot_features_final")
    ap.add_argument("--use-cuda", action="store_true")
    args = ap.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    Xs, y = [], []
    meta = {"layers": args.layers, "examples": []}
    Path(args.out).mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    for rec in load_jsonl(args.data):
        # Take ONLY the last line as the final answer sentence
        final_line = rec["answer"].splitlines()[-1].strip()
        if not final_line:
            n_skipped += 1
            continue
        vec = span_mean_vecs(model, tok, rec["prompt"], final_line, device, args.layers)
        Xs.append(vec)
        y.append(int(rec["label"]))
        meta["examples"].append({
            "mode": rec["mode"], "question": rec["question"],
            "final_line": final_line
        })

    X = np.vstack(Xs)
    y = np.array(y, dtype=int)
    np.save(Path(args.out)/"X.npy", X)
    np.save(Path(args.out)/"y.npy", y)
    with open(Path(args.out)/"meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved features {X.shape} to {args.out} (skipped {n_skipped} empty finals)")

if __name__ == "__main__":
    main()
