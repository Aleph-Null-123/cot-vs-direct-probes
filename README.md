# CoT vs Direct Probes

This project investigates whether small language models (GPT-2 and distilgpt2) encode a latent "reasoning mode" when prompted with Chain-of-Thought (CoT) versus Direct answering.

## Summary
- **Linear probes** on hidden states can reliably separate CoT vs Direct final sentences (AUROC ≈ 0.95).  
- **Lexical baselines** (TF-IDF) achieve equivalent performance, showing the effect is explained by surface style.  
- **Entropy** is weaker; **length** is near chance.  
- The signal **localizes to mid–late layers** in both GPT-2 and distilgpt2.  
- **Causal steering** along the probe direction fails, indicating the representation is *readable but not steerable* under simple linear interventions.


## Reproduction
The repo contains scripts for:
- Feature extraction (`extract_features_finalline.py`)  
- Probe training (`train_probe_group.py`)  
- Baselines (lexical, length, entropy)  
- Causal intervention (`causal_steer.py`)  
- Plotting (`aggregate_and_plot.py`, `plot_steering_effect.py`)  

Example run (GPT-2):
```bash
python cot_extract_features_finalline.py --data cot_outputs.jsonl --model gpt2 --layers 3 6 9 12 --out cot_features_final
python cot_train_probe_group.py --feat_dir cot_features_final --hidden_dim 768 --per_layer
