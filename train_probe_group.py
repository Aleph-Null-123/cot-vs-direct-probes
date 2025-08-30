#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

def run_groupcv(X, y, groups, k=5):
    gkf = GroupKFold(n_splits=k)
    accs, aucs = [], []
    for tr, te in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:,1]
        pred = (prob>=0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs)), float(np.std(aucs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="cot_features_final")
    ap.add_argument("--hidden_dim", type=int, default=768)  # GPT-2
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--out", default="cot_probe_group_metrics.json")
    ap.add_argument("--label_shuffle", action="store_true", help="sanity: shuffle labels before CV")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    X = np.load(feat_dir/"X.npy")
    y = np.load(feat_dir/"y.npy")
    meta = json.load(open(feat_dir/"meta.json"))
    layers = meta["layers"]
    questions = [e["question"] for e in meta["examples"]]

    # map question -> id for grouping
    q_to_id = {q:i for i,q in enumerate(sorted(set(questions)))}
    groups = np.array([q_to_id[q] for q in questions])

    if args.label_shuffle:
        rng = np.random.default_rng(0)
        y = rng.permutation(y)

    results = {}
    if args.per_layer:
        for i, li in enumerate(layers):
            Xi = X[:, i*args.hidden_dim:(i+1)*args.hidden_dim]
            acc_m, acc_s, auc_m, auc_s = run_groupcv(Xi, y, groups, k=5)
            results[f"layer_{li}"] = {"acc_mean": acc_m, "acc_std": acc_s,
                                      "auc_mean": auc_m, "auc_std": auc_s}

    acc_m, acc_s, auc_m, auc_s = run_groupcv(X, y, groups, k=5)
    results["concat_layers"] = {"acc_mean": acc_m, "acc_std": acc_s,
                                "auc_mean": auc_m, "auc_std": auc_s,
                                "layers": layers, "n": int(len(y))}
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
