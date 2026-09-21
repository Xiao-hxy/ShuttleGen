# ShuttleGen —— Badminton Landing Point Prediction as Conditional Distribution Estimation

Reformulating badminton shot landing prediction from **point regression** to **conditional distribution estimation**.

## Task Definition

Given the first *t* shots of a rally (shot type, landing coordinates, player positions, player IDs, score, ...), predict a **full probability distribution** over the landing position of shot *t+1*:

$$p(x, y \mid \text{rally context})$$

This is fundamentally different from the standard point-regression formulation (single $(x, y)$ output), because at any given game situation, **multiple landing zones are equally reasonable** (e.g., smash left vs. drop right).

## Core Insight

Point-regression models trained with MSE are forced to output the **mean** of all plausible landing positions — mathematically, this is the behavior of MSE loss under multi-modality. The result is predictions that systematically fall **between** real landing zones, rather than on any of them. Distribution estimation naturally handles this.

## Models

All versions share the **same GRU sequence encoder** — differences are purely in the **output head / routing mechanism**, enabling clean controlled-variable comparison.

| Version | Core Change |
|---|---|
| **v1** `baseline_v1.py` | Point-regression baseline (GRU → MLP, single $(x,y)$ output) |
| **v2** `mdn_v2.py` | Mixture Density Network — replace head with 5-component Gaussian mixture |
| **v3** `moe_v3.py` | Two-stage Mixture-of-Experts — type gate ("what shot?") × landing experts ("where lands?") |
| **v4** `tgmm_v4.py` | Truncated Gaussian Mixture Model — v3 + court-boundary truncation correction for NLL |
| **v4b** `tgmm_v4b_tight.py` | Ablation: same architecture, tighter truncation boundaries |

### Uncertainty Baselines

| File | Description |
|---|---|
| `deep_ensemble.py` | 5× v1 trained independently, uniform-weighted Gaussian mixture |
| `mc_dropout.py` | v1 + 20 dropout forward passes, residual-estimated uncertainty |

## Results (ShuttleSet22, test split)

All numbers use the **same GRU encoder**, **same protocol**, **same test set**. Coordinates in centimeters (cm).

| Model | MAE ↓ | minADE-5 ↓ | Cov@90% | NLL ↓ |
|---|---|---|---|---|
| Constant baseline | 127.88 | — | — | — |
| v1 GRU point regression | 76.78 | — | — | — |
| Deep Ensembles (5×v1) | **76.47** | 64.62 | 0.943 | 2.148 |
| MC Dropout (20×) | 76.76 | 70.58 | 0.924 | 2.168 |
| v2 MDN | 77.92 | 39.56 | 0.863 | 1.377 |
| v3 Two-stage MoE | 77.10 | 40.08 | **0.897** | 1.316 |
| **v4 TGMM (ours)** | 76.91 | **38.64** | 0.895 | **1.211** |
| v4 oracle (diagnostic upper bound) | (50.10) | (26.64) | (0.877) | (0.657) |

**Notes:**
- **MAE** = mean((\|Δx\| + \|Δy\|)/2), real court coordinates (cm)
- **minADE-5** = sample 5 hypotheses, take the best one, average MAE over all samples — strict test of multi-modality coverage
- **Cov@90%** = fraction of true landings falling within the model's 90% highest-density region (ideal = 0.90)
- **NLL** = Negative Log-Likelihood in normalized coordinate space (constants cancel across models)
- **Oracle row**: not a submitted model — assumes perfect shot-type routing, reveals the theoretical architecture upper bound

## Key Findings

1. **Point accuracy stagnation**: Every method (v1/v2/v3/v4/DE/MC) achieves MAE between 76-78 cm — the "best possible point" on this task is saturated, consistent with 16 teams on ShuttleSet22 all clustering at normalized MAE ≈ 0.70.

2. **Distribution quality gap**: v4 TGMM cuts NLL by **44%** vs. Deep Ensembles (1.211 vs. 2.148) and minADE-5 by **40%** (38.64 vs. 64.62). DE and MC Dropout achieve good MAE but catastrophically bad distribution quality — confirming that multi-modality must be modeled from the start, not patched on top of point regression.

3. **Oracle analysis**: v4 oracle MAE drops to **50.10 cm** (-34.7%), proving the architecture's full potential. The bottleneck is 100% in the **type gate** (top-1 accuracy only 54.6%), not in the landing experts themselves.

## Usage

```bash
conda create -n badminton_cv python=3.11
conda activate badminton_cv
pip install -r requirements.txt
```

Preprocess the data pipeline (requires ShuttleSet22 raw data):
```bash
python data_pipeline.py
```

Train each version:
```bash
python baseline_v1.py      # ~5 min on CPU
python mdn_v2.py
python moe_v3.py
python tgmm_v4.py
python deep_ensemble.py    # 5× v1 training runs
python mc_dropout.py
```

All outputs are saved to `results/` as JSON (metrics), `.pt` (weights, not tracked by git), and `.png` (visualizations).

## Project Structure

```
ShuttleGen/
├── data_pipeline.py          # v0 — raw CSV → train/val/test .npz
├── baseline_v1.py            # v1 — GRU point regression
├── mdn_v2.py                 # v2 — Mixture Density Network head
├── moe_v3.py                 # v3 — Two-stage Mixture-of-Experts
├── tgmm_v4.py                # v4 — Truncated Gaussian Mixture Model
├── tgmm_v4b_tight.py         # v4b — Boundary ablation
├── deep_ensemble.py          # Uncertainty baseline
├── mc_dropout.py             # Uncertainty baseline
├── .gitignore
├── requirements.txt
└── results/
    └── *.json                # Metrics only (weights/data/images gitignored)
```

## Data

This repository contains **code only**, not data. We use the [ShuttleSet22](https://github.com/CoachAI-Projects/CoachAI-Challenge-IJCAI2023) dataset — 58 top-level badminton matches, 44k labeled strokes, released as part of the IJCAI 2023 CoachAI Challenge. Preprocessing rules (train/val/test split, stroke-type merging, homography projection, z-score normalization) are all in `data_pipeline.py`.
