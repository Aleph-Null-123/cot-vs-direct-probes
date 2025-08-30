#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--feat_dir", default="cot_features_final")
ap.add_argument("--out", default="length_baseline_metrics.json")
args = ap.parse_args()

meta = json.load(open(Path(args.feat_dir)/"meta.json"))
y, lengths, questions = [], [], []
for e in meta["examples"]:
    final_line = e["final_line"]
    y.append(1 if e["mode"]=="cot" else 0)
    lengths.append(len(final_line.split()))
    questions.append(e["question"])

y = np.array(y); X = np.array(lengths).reshape(-1,1)
q_to_id = {q:i for i,q in enumerate(sorted(set(questions)))}
groups = np.array([q_to_id[q] for q in questions])

gkf = GroupKFold(n_splits=5)
accs, aucs = [], []
for tr, te in gkf.split(X, y, groups=groups):
    clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:,1]
    accs.append(accuracy_score(y[te], (p>=0.5)))
    aucs.append(roc_auc_score(y[te], p))

res = {"length_acc_mean": float(np.mean(accs)), "length_auroc_mean": float(np.mean(aucs))}
print(res)
json.dump(res, open(args.out, "w"), indent=2)
