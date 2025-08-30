#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def run_cv(X, y, k=5):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    accs, aucs = [], []
    for tr, te in kf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:,1]
        pred = (prob>=0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs)), float(np.std(aucs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="cot_features")
    ap.add_argument("--hidden_dim", type=int, default=768, help="d_model of your network (GPT2=768; T5-small=512)")
    ap.add_argument("--per_layer", action="store_true", help="evaluate each layer slice separately")
    ap.add_argument("--out", default="cot_probe_metrics.json")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    X = np.load(feat_dir/"X.npy"); y = np.load(feat_dir/"y.npy")
    meta = json.load(open(feat_dir/"meta.json"))
    layers = meta["layers"]

    results = {}
    if args.per_layer:
        # split concatenated features into per-layer chunks
        for i, li in enumerate(layers):
            Xi = X[:, i*args.hidden_dim:(i+1)*args.hidden_dim]
            acc_m, acc_s, auc_m, auc_s = run_cv(Xi, y, k=5)
            results[f"layer_{li}"] = {"accuracy_mean": acc_m, "accuracy_std": acc_s,
                                      "auroc_mean": auc_m, "auroc_std": auc_s}
    # all layers concatenated
    acc_m, acc_s, auc_m, auc_s = run_cv(X, y, k=5)
    results["concat_layers"] = {"accuracy_mean": acc_m, "accuracy_std": acc_s,
                                "auroc_mean": auc_m, "auroc_std": auc_s,
                                "layers": layers, "n": int(len(y))}
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
