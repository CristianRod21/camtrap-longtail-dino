# Cycle 4 — Partial Fine-Tuning of DINOv2

**Research question**: Cycle 3 showed that no classifier-level intervention on
*frozen* DINOv2 features (cRT, MLP, focal, NCM) closes the seen→unseen
generalization gap. If we partially adapt the representation itself by
unfreezing the last K transformer blocks, does unseen-camera performance
improve? Is there a dose-response over K?

---

## Experimental design

Dose-response over the number of unfrozen transformer blocks **K ∈ {0, 2, 4}**.
Everything else held constant vs. Cycle 1–3: same backbone, same data, same
geographic split. K=0 is exactly the Cycle 3 Linear CE baseline (no re-run
needed).

| K | Trainable | Head | Loss | Notes |
|---|---|---|---|---|
| 0 | 0.16M (head only) | BN + Linear | CE | Same as Cycle 3 Linear CE baseline — not re-run here |
| 2 | ~14.18M backbone + 0.16M head | BN + Linear | CE | First "release the representation" condition |
| 4 | ~28M backbone + 0.16M head | BN + Linear | CE | Higher dose |

Training recipe (fixed, no config file — hardcoded in the script to mirror
the original notebook exactly):

- AdamW, discriminative LRs: head fixed at `1e-3`, backbone swept over
  `{5e-6, 1e-5, 2e-5, 5e-5, 1e-4}` (5 LRs × 2 K = 10 runs).
- Cosine schedule with 500-step linear warmup.
- Weight decay 0.05 on backbone, 0.0 on head.
- AMP, batch size 32, 5 epochs, model selection by best val macro-F1.
- Linear head + plain CE only — deliberately, to isolate the representation
  effect from classifier-side or loss-side confounds already explored in
  Cycle 3.

This script trains end-to-end each epoch (no frozen-feature caching), so the
augmentation-disable fix used in Cycles 1–3 does not apply here — train-time
`RandomHorizontalFlip` is real augmentation in this setup, not a frozen
one-off baked into a cache.

---

## Model

Single backbone: `dinov2_vitb14`. No YAML config — the script hardcodes the
dataset root/metadata path and hyperparameters directly (matching the
upstream notebook this was ported from).

---

## How to run

Run from the project root. Each K uses a distinct `--checkpoint_dir` group
(`k{K}_lr{lr}/best.ckpt`), and the script resumes — it skips any run whose
`best.ckpt` already exists.

```bash
# Sweep K=2 (5 backbone LRs x 5 epochs)
uv run src/cycle4_partial_unfreeze.py --unfreeze_k 2 \
    --wandb_project iwildcam2022_cycle4_sweep \
    --checkpoint_dir checkpoints/cycle4_sweep

# Sweep K=4
uv run src/cycle4_partial_unfreeze.py --unfreeze_k 4 \
    --wandb_project iwildcam2022_cycle4_sweep \
    --checkpoint_dir checkpoints/cycle4_sweep
```

Checkpoints land in `checkpoints/cycle4_sweep/k{K}_lr{lr}/best.ckpt`, each
containing `state_dict`, `config`, `best_val_f1`, `test_f1_macro`, and
per-epoch logs.

---

## Out of scope for this cycle

The earlier iteration of this pipeline also had a q10 follow-up (cRT trained
on frozen features extracted from the best K=4 checkpoint) and a q11
PCA-interpretability notebook. Both are analysis/exploration built on top of
the K-sweep checkpoints, not the training piece itself — not ported here.
