# Validation — WILDS iWildCam (external benchmark)

**Goal**: test whether the cycle 4 winner (DINOv2 ViT-B/14, last K=4 blocks
fine-tuned end-to-end, CE) transfers to a second, independent context —
the official WILDS iWildCam benchmark, with its own published train/val/test
split and a harder OOD-location shift than our custom split. This is not a
new design cycle: hyperparameters are taken verbatim from cycles 1/3/4, not
re-tuned for WILDS.

---

## Conditions

Three conditions, evaluated on all four WILDS eval splits (`id_val`,
`id_test`, `val` = OOD val, `test` = OOD test):

| # | Condition | Backbone | Head | Notes |
|---|---|---|---|---|
| 1 | **Frozen Linear CE** | DINOv2 ViT-B/14 (frozen) | BN + Linear | Pre-fine-tuning reference. Cycle 1/3 head recipe (13-LR sweep, SGD-momentum-cosine, batch 256, 15 epochs). |
| 2 | **K=4 + CE** | DINOv2 ViT-B/14 (last 4 blocks + final norm) | BN + Linear | Cycle 4 winner verbatim (`bb_lr=2e-5`, `head_lr=1e-3`, AdamW, 500-step warmup, cosine, AMP, batch 32, 5 epochs). |
| 3 | **K=4 + cRT** | DINOv2 ViT-B/14 (K=4 backbone, re-frozen) | BN + Linear | Cycle 4 q10 recipe verbatim: re-extract CLS features from the trained K=4 backbone, then 13-LR sweep x 10 epochs, batch 256, class-balanced sampler. |

Model selection for all three is on **WILDS OOD val (`val`)** macro-F1,
matching the WILDS leaderboard convention.

---

## Data

WILDS iWildCam (v2.0) auto-downloads on first run via the `wilds` package
(`download=True` in `src/data/wilds.py`) to `WILDS_ROOT` (set that constant
to your local path) if not already present there (~11GB compressed, ~12GB
extracted). No manual setup step needed — the `wilds` dependency is
declared in `pyproject.toml`.

---

## How to run

Run from the project root, in order — each step's output feeds the next.
All scripts are resume-safe within a run (they skip work whose output
already exists) but there's no cross-run checkpointing beyond that.

```bash
# 0. Preflight: frozen DINOv2 features for all 5 WILDS splits, then
#    frequency segments + per-split eligible classes (both skip if cached)
uv run src/validation/00_preflight.py

# 1. Condition 1: frozen Linear CE — 13-LR sweep on cached features
uv run src/validation/01_frozen_linear_ce.py

# 2. Condition 2: K=4 + CE end-to-end training
uv run src/validation/02_k4_ce_train.py

# 3. Condition 3: K=4 + cRT — extracts frozen K=4 features (skips if
#    cached), then 13-LR sweep on them
uv run src/validation/03_k4_crt.py

# 4. Build descriptive table, paired bootstrap, leaderboard context table
uv run src/validation/04_evaluate_and_report.py
```

Checkpoints and results land under `checkpoints/validation/` (gitignored,
same convention as `checkpoints/cycle{1..4}_sweep/`):

- `checkpoints/validation/{frozen_linear_ce,k4_ce,k4_crt}/` — best
  head/checkpoint + per-split predictions per condition
- `checkpoints/validation/results/` — `wilds_class_segments.json`,
  `descriptive_metrics.csv`, `paired_bootstrap.csv`,
  `leaderboard_context.csv`, `per_class_f1/`, `bootstrap_samples/`

---

## Statistical analysis

- Two paired comparisons — K=4+CE vs Frozen Linear CE (main effect),
  K=4+cRT vs K=4+CE (cRT interaction) — Holm-Bonferroni corrected across
  the two **Total** p-values within each test split (`id_test`, `test`).
- Cluster bootstrap by camera (`location`), B=2000, cameras with <15 images
  dropped.
- Frequency segments (Rare/Common/Frequent) from **WILDS train counts**, not
  iWildCam 2022's. Per-segment p-values stay raw/descriptive.

---

## Out of scope

No K=2, no MLP heads, no focal loss, no Serengeti, no sequence aggregation,
no location-prior post-hoc adjustment, no leaderboard-chasing writeup. The
point is to test the cycle-4 configuration in a second context, not to
re-design it for WILDS.
