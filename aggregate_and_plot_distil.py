#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

probe_json   = Path("cot_probe_group_metrics_distil.json")
lex_json     = Path("lexical_baseline_metrics_distil.json")
len_json     = Path("length_baseline_metrics_distil.json")
ent_json     = Path("entropy_baseline_metrics_distil.json")

probe = json.loads(probe_json.read_text())
lex   = json.loads(lex_json.read_text())
lng   = json.loads(len_json.read_text())
ent   = json.loads(ent_json.read_text())

# -------- Bar: AUROC by method --------
probe_auroc = probe["concat_layers"]["auc_mean"]
lex_auroc   = lex["lexical_auroc_mean"]
ent_auroc   = ent["entropy_auroc_mean"]
len_auroc   = lng["length_auroc_mean"]

methods = ["Probe", "Lexical", "Entropy", "Length"]
vals    = [probe_auroc, lex_auroc, ent_auroc, len_auroc]

plt.figure(figsize=(6.5,4.2))
plt.bar(methods, vals, edgecolor="black")
plt.axhline(0.5, linestyle="--", linewidth=1)
for i, v in enumerate(vals):
    plt.text(i, min(v+0.03, 0.98), f"{v:.2f}", ha="center", fontsize=10)
plt.ylim(0,1)
plt.ylabel("AUROC")
plt.title("CoT vs Direct — AUROC by Method (distilgpt2, final line)")
plt.tight_layout()
plt.savefig("auroc_comparison_distil.png", dpi=300)
plt.close()

# -------- Line: Probe AUROC across layers --------
layers = probe["concat_layers"]["layers"]  # e.g., [2,4,6]
layer_aurocs = [probe[f"layer_{li}"]["auc_mean"] for li in layers]

plt.figure(figsize=(6.5,4.2))
plt.plot(layers, layer_aurocs, marker="o", linewidth=2)
for x, y in zip(layers, layer_aurocs):
    plt.text(x, min(y+0.02, 0.98), f"{y:.2f}", ha="center", fontsize=9)
plt.axhline(0.5, linestyle="--", linewidth=1)
plt.xlabel("Layer")
plt.ylabel("AUROC")
plt.title("Probe AUROC across distilgpt2 Layers (final line)")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("probe_auroc_by_layer_distil.png", dpi=300)
plt.close()

print("Saved: auroc_comparison_distil.png, probe_auroc_by_layer_distil.png")
