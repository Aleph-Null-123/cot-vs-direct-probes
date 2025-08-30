#!/usr/bin/env bash
set -euo pipefail

# --------- Config ---------
MODEL="distilgpt2"
LAYERS=(2 4 6)           # distilgpt2 has 6 layers; we'll sample these
FEAT_DIR="cot_features_final_distil"
OUT_PROBE_JSON="cot_probe_group_metrics_distil.json"
OUT_LEX_JSON="lexical_baseline_metrics_distil.json"
OUT_LEN_JSON="length_baseline_metrics_distil.json"
OUT_ENT_JSON="entropy_baseline_metrics_distil.json"
DATA_JSONL="cot_outputs.jsonl"   # reuse your existing final-line dataset

# --------- 1) Feature extraction (final line only) ---------
echo "=== [1/5] Extracting final-line features for $MODEL on layers: ${LAYERS[*]} ==="
python extract_features_finalline.py \
  --data "$DATA_JSONL" \
  --model "$MODEL" \
  --layers "${LAYERS[@]}" \
  --out "$FEAT_DIR"

# --------- 2) Probe (GroupKFold by question), per-layer + concat ---------
echo "=== [2/5] Training GroupKFold probe (hidden_dim=768) ==="
python train_probe_group.py \
  --feat_dir "$FEAT_DIR" \
  --hidden_dim 768 \
  --per_layer \
  --out "$OUT_PROBE_JSON"

# --------- 3) Baselines (lexical, length, entropy) ---------
echo "=== [3/5] Running lexical baseline ==="
python lexical_baseline_final.py --feat_dir "$FEAT_DIR" --out "$OUT_LEX_JSON"

echo "=== [3/5] Running length baseline ==="
python length_baseline_final.py  --feat_dir "$FEAT_DIR" --out "$OUT_LEN_JSON"

echo "=== [3/5] Running entropy baseline (model=$MODEL) ==="
python entropy_baseline_final.py --feat_dir "$FEAT_DIR" --model "$MODEL" --out "$OUT_ENT_JSON"

# --------- 4) Aggregation + Plots (writes 2 PNGs) ---------
echo "=== [4/5] Creating aggregator/plotter ==="
cat > aggregate_and_plot_distil.py << 'PY'
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
PY

chmod +x aggregate_and_plot_distil.py
echo "=== [5/5] Generating plots ==="
python aggregate_and_plot_distil.py

echo "Done. Plots:
 - auroc_comparison_distil.png
 - probe_auroc_by_layer_distil.png
JSON outputs:
 - $OUT_PROBE_JSON
 - $OUT_LEX_JSON
 - $OUT_LEN_JSON
 - $OUT_ENT_JSON"
