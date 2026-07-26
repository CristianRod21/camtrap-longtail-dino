# Cycle 3 — Classification Strategies for Long-Tailed Distributions

**Research question**: Given frozen DINOv2 ViT-B/14 features (full images, no cropping — same setup as Cycle 1), is the classifier the bottleneck for rare-class performance? What combination of classifier architecture and training strategy best handles the long-tailed class distribution?

---

## Experimental design

Four core strategies compared against the Linear CE baseline, plus one optional strategy testing the interaction between classifier expressivity and loss rebalancing.

| # | Strategy | Classifier | Loss | Sampling | Sweep | Rationale |
|---|---|---|---|---|---|---|
| 1 | **Linear CE** | BN + Linear | CE | natural | 13 LRs | Baseline (same as Cycle 1) |
| 2 | **cRT** | BN + Linear | CE | balanced | 13 LRs | Strongest strategy for frozen representations (Kang et al., 2020) |
| 3 | **NCM** | — | — | — | 1 | Zero-shot extreme: single centroid per class, no training |
| 4 | **MLP CE** | BN + MLP | CE | natural | 13 LRs | Tests classifier expressivity |
| 5 | **MLP Focal** | BN + MLP | Focal | natural | 4 gammas × 13 LRs | Expressivity + rebalancing interaction (optional) |

Model selection uses **val macro-F1** (not accuracy) to favor rare classes.

### Why these strategies

- **cRT** (Kang et al., 2020, "Decoupling"): the reference strategy for decoupled training on frozen representations. Balanced sampling re-calibrates the classifier without modifying feature learning.
- **NCM**: the conceptual extreme for rare classes — a single centroid defines the class. Tests whether DINOv2 features are separable enough that no learning is needed.
- **MLP CE**: tests whether the linear probe underutilizes feature expressivity. A non-linear decision boundary may capture structure the linear head cannot.
- **MLP Focal** (optional): if MLP alone improves, does adding focal loss (which down-weights easy examples) further help rare classes?

### MLP architecture

Single hidden layer: `BN(768) → Linear(768, 256) → ReLU → Linear(256, C)`.

Minimal non-linear extension of the linear probe — no dropout, no extra depth, to keep the comparison to Linear CE clean. BatchNorm follows the standard protocol for frozen-feature evaluation (Park et al., 2023). The single-hidden-layer design follows Chen et al. (2019), "A Closer Look at Few-Shot Classification."

---

## Models

| Config | Input |
|---|---|
| `cycle_3_thesis/dinov2_vitb14.yaml` | Full frame (same as Cycle 1) |
| `cycle_3_thesis/dinov2_vitb14_cropped.yaml` | MegaDetector crop (same as Cycle 2) — not run yet, kept for a future crop-vs-strategy interaction check |

---

## How to run

Run from the project root. Features are cached to disk on first run (`checkpoints/cycle3_sweep/cached_features_<backbone>.pt`); subsequent runs load from disk instantly. Each strategy uses a distinct `--checkpoint_dir` group, so re-running one strategy doesn't touch another's checkpoints.

```bash
# Run a single strategy:
uv run src/cycle3_sweep.py \
    --config src/configs/model/cycle_3_thesis/dinov2_vitb14.yaml \
    --strategy ce \
    --wandb_project iwildcam2022_cycle3_sweep \
    --checkpoint_dir checkpoints/cycle3_sweep

# Available strategies: ce, crt, ncm, mlp_ce, mlp_focal, all
```

The script supports resume — it skips LR/strategy combinations where a checkpoint already exists, so it's safe to re-run after an interruption.

Results land in `checkpoints/cycle3_sweep/results_<strategy>.yaml`.

---

## Statistical analysis plan

- 4 paired comparisons against the Linear CE baseline (5 if MLP Focal is included)
- Holm-Bonferroni correction for multiple comparisons
- Cluster bootstrap (resample cameras) for confidence intervals — accounts for spatial dependence, same methodology as Cycle 1's `bootstrap_methodology.ipynb`
- Frequency segments (rare/common/frequent) as descriptive, not inferential

---

## Out of scope for this cycle

The earlier iteration of this pipeline also had exploratory follow-ups beyond the 5-strategy comparison above (feature-separability diagnostics, prototype classifiers, sequence-level aggregation, WILDS benchmark evaluation). Those weren't ported — this cycle is scoped to the core strategy sweep only.
