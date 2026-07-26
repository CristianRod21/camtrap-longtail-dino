"""Condition 3 — K=4 + cRT head on frozen K=4 features.

Two steps in one script, since the K=4-backbone feature cache has exactly
one consumer (this script's own head sweep) — no reason to serialize it
through a separate file the way the shared frozen-features cache in
`00_preflight.py` does.

1. Extract frozen CLS features from the trained K=4 backbone (skips if
   already cached).
2. cRT protocol verbatim from cycle 4: BN+Linear head, class-balanced
   WeightedRandomSampler (1/class_count), SGD-momentum-cosine, batch 256,
   10 epochs, lr scaled by batch/256, 13-LR sweep. Model selection by OOD
   val (`val`) eligible-macro-F1.
"""

from __future__ import annotations

import gc
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
    build_partial_unfreeze_model,
    eval_head,
    load_segments,
    make_head,
    train_head_on_features,
)

from data.wilds import eval_transform, get_loader

UNFREEZE_K = 4
EXTRACT_BATCH_SIZE = 64
USE_AMP = True
EXTRACT_NUM_WORKERS = 2  # keep low: each loader spawns workers + a WILDS dataset


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            f = model.extract_features(imgs)
        feats.append(f.float().cpu())
        labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


def extract_k4_features(feat_file) -> None:
    if feat_file.exists():
        size_mb = feat_file.stat().st_size / 1e6
        print(f"[skip] {feat_file} already present ({size_mb:.1f} MB)")
        return

    in_ckpt = CHECKPOINT_DIR / "k4_ce" / "best.ckpt"
    if not in_ckpt.exists():
        raise FileNotFoundError(f"K=4 checkpoint not found at {in_ckpt}; run 02 first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    torch.manual_seed(SEED)

    print(f"Loading K=4 backbone from {in_ckpt}…")
    model, n_blocks = build_partial_unfreeze_model(NUM_CLASSES, unfreeze_k=UNFREEZE_K)
    meta = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(meta["state_dict"])
    print(f"  loaded ckpt; best_val_f1_eligible={meta.get('best_val_f1_eligible'):.4f}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    cached = {"num_classes": NUM_CLASSES, "source_ckpt": str(in_ckpt)}
    for split in ("train", "id_val", "id_test", "val", "test"):
        print(f"Extracting {split}…", flush=True)
        loader = get_loader(
            split,
            batch_size=EXTRACT_BATCH_SIZE,
            shuffle=False,
            transform=eval_transform(),
            num_workers=EXTRACT_NUM_WORKERS,
            pin_memory=False,
        )
        t0 = time.time()
        feats, labels = extract(model, loader, device)
        print(
            f"  {split}: feats={tuple(feats.shape)} labels={tuple(labels.shape)} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        cached[f"{split}_features"] = feats
        cached[f"{split}_labels"] = labels

        del loader
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(cached, feat_file)
    print(f"[save] {feat_file}  ({feat_file.stat().st_size / 1e6:.1f} MB)")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    feat_file = CHECKPOINT_DIR / "k4_ce" / "cached_features_wilds_k4ft.pt"
    extract_k4_features(feat_file)

    out_dir = CHECKPOINT_DIR / "k4_crt"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"Loading K=4 features from {feat_file}…")
    cached = torch.load(feat_file, map_location="cpu", weights_only=True)
    train_features = cached["train_features"]
    train_labels = cached["train_labels"]
    val_features = cached["val_features"]  # OOD val
    val_labels = cached["val_labels"]
    print(f"  train={tuple(train_features.shape)}  val={tuple(val_features.shape)}")

    segs = load_segments()
    elig_val = set(segs["eligible_per_split"]["val"])
    elig_per_split = {s: set(segs["eligible_per_split"][s]) for s in EVAL_SPLITS}

    print("\nSweeping 13 LRs (10 epochs, batch 256, class-balanced sampler)…")
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
            epochs=10,
            balanced=True,
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
        elig = elig_per_split[split]
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
