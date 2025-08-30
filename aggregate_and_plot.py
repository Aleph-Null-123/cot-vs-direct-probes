#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Where your JSONs live
probe_json      = Path("cot_probe_group_metrics.json")               # from cot_train_probe_group.py
lexical_json    = Path("lexical_baseline_metrics.json")              # from lexical_baseline_final.py
length_json     = Path("length_baseline_metrics.json")               # from length_baseline_final.py
entropy_json    = Path("entropy_baseline_metrics.json")              # from entropy_baseline_final.py

# ---- Load metrics ----
probe = json.load(open(probe_json))
lex   = json.load(open(lexical_json))
lng   = json.load(open(length_json))
ent   = json.load(open(entropy_json))

# AUROC numbers
probe_auroc = probe["concat_layers"]["auc_mean"]
lex_auroc   = lex["lexical_auroc_mean"]
ent_auroc   = ent["entropy_auroc_mean"]
len_auroc   = lng["length_auroc_mean"]

# ---- Plot 1: AUROC comparison bar chart ----
methods = ["Probe", "Lexical", "Entropy", "Length"]
vals = [probe_auroc, lex_auroc, ent_auroc, len_auroc]

plt.figure(figsize=(6,4))
bars = plt.bar(methods, vals, edgecolor="black")
plt.axhline(0.5, linestyle="--", linewidth=1)
plt.ylabel("AUROC")
plt.title("CoT vs Direct — AUROC by Method (final line only)")
plt.ylim(0,1)
for i,v in enumerate(vals):
    plt.text(i, min(v+0.03,0.97), f"{v:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("auroc_comparison.png", dpi=300)
plt.close()

print("Saved auroc_comparison.png")

# ---- Plot 2: Probe AUROC across layers ----
layers = probe["concat_layers"]["layers"]  # e.g., [3,6,9,12]
layer_aurocs = []
for li in layers:
    key = f"layer_{li}"
    if key in probe:
        layer_aurocs.append(probe[key]["auc_mean"])
    else:
        layer_aurocs.append(None)

plt.figure(figsize=(6,4))
plt.plot(layers, layer_aurocs, marker="o", linewidth=2)
plt.axhline(0.5, linestyle="--", linewidth=1)
plt.xlabel("Layer")
plt.ylabel("AUROC")
plt.title("Probe AUROC across GPT-2 Layers (final line)")
for x,y in zip(layers, layer_aurocs):
    if y is not None:
        plt.text(x, min(y+0.02,0.98), f"{y:.2f}", ha="center", fontsize=9)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("probe_auroc_by_layer.png", dpi=300)
plt.close()

print("Saved probe_auroc_by_layer.png")
