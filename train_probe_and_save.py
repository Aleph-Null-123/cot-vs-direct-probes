#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def train_and_save_layer(X, y, d, out_dir, layer_tag):
    # CV for reporting
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs, accs = [], []
    for tr, te in kf.split(X, y):
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], (p>=0.5)))
    # Train on all data (the one we save)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    clf.fit(X, y)
    w = clf.coef_.reshape(d)   # [d_model]
    b = float(clf.intercept_[0])
    np.savez(Path(out_dir)/f"probe_{layer_tag}.npz", w=w, b=b, d_model=d)
    return float(np.mean(aucs)), float(np.mean(accs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="cot_features_final")
    ap.add_argument("--hidden_dim", type=int, default=768)   # GPT-2 small
    ap.add_argument("--out_dir", default="saved_probes")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    X = np.load(feat_dir/"X.npy")       # shape [N, L*d_model]
    y = np.load(feat_dir/"y.npy").astype(int)
    meta = json.load(open(feat_dir/"meta.json"))
    layers = meta["layers"]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    metrics = {"layers": layers, "hidden_dim": args.hidden_dim, "n": int(len(y)), "per_layer": {}}

    for i, li in enumerate(layers):
        Xi = X[:, i*args.hidden_dim:(i+1)*args.hidden_dim]
        auc, acc = train_and_save_layer(Xi, y, args.hidden_dim, args.out_dir, f"layer{li}")
        metrics["per_layer"][str(li)] = {"auroc_cv": auc, "acc_cv": acc, "weight_file": f"probe_layer{li}.npz"}

    with open(Path(args.out_dir)/"probe_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
