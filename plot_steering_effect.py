#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

data = [json.loads(l) for l in open(Path("steer_results.jsonl"), "r", encoding="utf-8")]

by_alpha_p, by_alpha_log = defaultdict(list), defaultdict(list)
for r in data:
    by_alpha_p[r["alpha"]].append(r["probe_p_cot"])
    by_alpha_log[r["alpha"]].append(r["probe_logit"])

alphas = sorted(by_alpha_p.keys())
mean_p = [float(np.mean(by_alpha_p[a])) for a in alphas]
std_p  = [float(np.std(by_alpha_p[a])) for a in alphas]
mean_l = [float(np.mean(by_alpha_log[a])) for a in alphas]
std_l  = [float(np.std(by_alpha_log[a])) for a in alphas]

# Plot p(CoT)
plt.figure(figsize=(6,4))
plt.plot(alphas, mean_p, marker="o", linewidth=2)
plt.fill_between(alphas, np.array(mean_p)-np.array(std_p), np.array(mean_p)+np.array(std_p), alpha=0.2)
plt.axhline(0.5, linestyle="--", linewidth=1)
plt.xlabel("α (steering strength)")
plt.ylabel("Mean probe p(CoT)")
plt.title("Causal steering along probe direction — probability")
plt.tight_layout()
plt.savefig("steering_curve_p.png", dpi=300)
plt.close()
print("Saved steering_curve_p.png")

# Plot logits (often more linear)
plt.figure(figsize=(6,4))
plt.plot(alphas, mean_l, marker="o", linewidth=2)
plt.fill_between(alphas, np.array(mean_l)-np.array(std_l), np.array(mean_l)+np.array(std_l), alpha=0.2)
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("α (steering strength)")
plt.ylabel("Mean probe logit")
plt.title("Causal steering along probe direction — logit")
plt.tight_layout()
plt.savefig("steering_curve_logit.png", dpi=300)
plt.close()
print("Saved steering_curve_logit.png")
