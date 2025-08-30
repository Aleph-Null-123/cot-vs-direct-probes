#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="cot_features_final")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--out", default="entropy_baseline_metrics.json")
    args = ap.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    meta = json.load(open(Path(args.feat_dir)/"meta.json"))
    finals = [e["final_line"] for e in meta["examples"]]
    labels = np.array([1 if e["mode"]=="cot" else 0 for e in meta["examples"]])
    questions = [e["question"] for e in meta["examples"]]
    q_to_id = {q:i for i,q in enumerate(sorted(set(questions)))}
    groups = np.array([q_to_id[q] for q in questions])

    def mean_next_token_entropy(text):
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**enc)
            logits = out.logits[0]
        seq = enc["input_ids"].shape[1]
        if seq < 2: return 0.0
        ents = []
        for i in range(seq-1):
            p = torch.softmax(logits[i], dim=-1)
            ents.append(float(-(p * torch.log(p + 1e-9)).sum().item()))
        return float(np.mean(ents))

    ent = np.array([mean_next_token_entropy(s) for s in finals], dtype=float).reshape(-1,1)

    gkf = GroupKFold(n_splits=5)
    accs, aucs = [], []
    for tr, te in gkf.split(ent, labels, groups=groups):
        clf = LogisticRegression(max_iter=1000).fit(ent[tr], labels[tr])
        p = clf.predict_proba(ent[te])[:,1]
        accs.append(accuracy_score(labels[te], (p>=0.5)))
        aucs.append(roc_auc_score(labels[te], p))

    res = {"entropy_acc_mean": float(np.mean(accs)), "entropy_auroc_mean": float(np.mean(aucs))}
    print(res)
    json.dump(res, open(args.out, "w"), indent=2)

if __name__ == "__main__":
    main()
