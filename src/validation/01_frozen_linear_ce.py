"""Condition 1 — DINOv2 frozen + Linear CE head.

13-LR sweep on cached WILDS features. BN+Linear head, SGD-momentum-cosine,
batch 256, 15 epochs. Model selection by OOD val (`val`) macro-F1 on the
per-split eligible class set. Saves the best head and the per-split
predictions for downstream evaluation.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from _common import (
    CHECKPOINT_DIR,
    EVAL_SPLITS,
    FEAT_DIM,
    LR_GRID,
    NUM_CLASSES,
    SEED,
    eval_head,
    load_cached_features,
    load_segments,
    make_head,
    train_head_on_features,
)


def main() -> None:
    out_dir = CHECKPOINT_DIR / "frozen_linear_ce"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("Loading cached WILDS features…")
    cached = load_cached_features()
    train_features = cached["train_features"]
    train_labels = cached["train_labels"]
    val_features = cached["val_features"]  # OOD val
    val_labels = cached["val_labels"]
    print(f"  train={tuple(train_features.shape)}  val={tuple(val_features.shape)}")

    segs = load_segments()
    elig_val = set(segs["eligible_per_split"]["val"])
    print(f"  OOD val eligible classes: {len(elig_val)}")

    print("\nSweeping 13 LRs (15 epochs, batch 256)…")
    sweep_results: dict[float, float] = {}
    best_state = None
    best_val_f1 = -1.0
    best_lr = None
    for lr in LR_GRID:
        t0 = time.time()
        state, val_f1 = train_head_on_features(
            train_features,
            train_labels,
            val_features,
            val_labels,
            val_eligible_classes=elig_val,
            num_classes=NUM_CLASSES,
            lr=lr,
            device=device,
            batch_size=256,
            epochs=15,
            balanced=False,
            seed=SEED,
        )
        sweep_results[lr] = val_f1
        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = state
            best_lr = lr
            marker = " *"
        print(f"  lr={lr:8.0e}  val_f1={val_f1:.4f}  ({time.time() - t0:.1f}s){marker}")

    print(f"\n[best] lr={best_lr:.0e}  val_f1={best_val_f1:.4f}")

    torch.save(
        {
            "state_dict": best_state,
            "best_lr": best_lr,
            "best_val_f1": best_val_f1,
            "sweep_results": sweep_results,
            "selection_split": "val",
            "selection_metric": "macro_f1_eligible",
        },
        out_dir / "best_head.pt",
    )

    head = make_head(FEAT_DIM, NUM_CLASSES).to(device)
    head.load_state_dict(best_state)
    head.eval()

    preds_by_split: dict[str, np.ndarray] = {}
    for split in EVAL_SPLITS:
        feats = cached[f"{split}_features"]
        lbls = cached[f"{split}_labels"]
        elig = set(segs["eligible_per_split"][split])
        preds, f1 = eval_head(head, feats, lbls, elig, device)
        preds_by_split[split] = preds
        print(f"  {split:8s} eligible-macro-F1={f1:.4f}  N={len(lbls)}")

    np.savez(
        out_dir / "predictions.npz",
        **{f"{split}_preds": preds_by_split[split] for split in EVAL_SPLITS},
    )
    print(f"\n[save] {out_dir / 'best_head.pt'}")
    print(f"[save] {out_dir / 'predictions.npz'}")


if __name__ == "__main__":
    main()
