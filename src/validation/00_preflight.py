"""Preflight: frozen DINOv2 features for all 5 WILDS splits + segments cache.

Both steps are idempotent (skip if their output already exists), so this is
safe to re-run. Two separate outputs because they're consumed differently
downstream: the feature cache is also read directly by
`01_frozen_linear_ce.py`, while the segments JSON is read by every
condition script and the final report.

Uses `unfreeze_k=0` (an all-frozen backbone) so feature extraction reuses
the same model-building code path as the K=4 conditions
(`build_partial_unfreeze_model`), instead of loading DINOv2 a second,
separate way.
"""

from __future__ import annotations

import gc
import json
import time
from collections import Counter

import torch
from _common import (
    CACHE_FILE,
    COMMON_THRESHOLD,
    EVAL_SPLITS,
    MIN_CAM_IMAGES,
    NUM_CLASSES,
    RARE_THRESHOLD,
    RESULTS_DIR,
    SEED,
    SEGMENTS_FILE,
    build_partial_unfreeze_model,
    load_cached_features,
)

from data.wilds import eval_transform, get_loader


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels, metadata = [], [], []
    for imgs, lbls, meta in loader:
        imgs = imgs.to(device, non_blocking=True)
        f = model.extract_features(imgs)
        feats.append(f.float().cpu())
        labels.append(lbls)
        metadata.append(meta)
    return torch.cat(feats), torch.cat(labels), torch.cat(metadata)


def extract_frozen_features() -> None:
    if CACHE_FILE.exists():
        size_mb = CACHE_FILE.stat().st_size / 1e6
        print(f"[skip] {CACHE_FILE} already present ({size_mb:.1f} MB)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    torch.manual_seed(SEED)

    model, _ = build_partial_unfreeze_model(NUM_CLASSES, unfreeze_k=0)
    model.eval().to(device)

    cached = {"num_classes": NUM_CLASSES}
    for split in ("train", "id_val", "id_test", "val", "test"):
        print(f"Extracting {split}…", flush=True)
        loader = get_loader(
            split,
            batch_size=32,
            shuffle=False,
            transform=eval_transform(),
            num_workers=4,
            drop_meta=False,
        )
        t0 = time.time()
        feats, labels, metadata = extract(model, loader, device)
        print(
            f"  {split}: feats={tuple(feats.shape)} labels={tuple(labels.shape)} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        cached[f"{split}_features"] = feats
        cached[f"{split}_labels"] = labels
        cached[f"{split}_metadata"] = metadata

        del loader
        gc.collect()
        torch.cuda.empty_cache()

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, CACHE_FILE)
    print(f"[save] {CACHE_FILE}  ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")


def build_segments_cache() -> None:
    if SEGMENTS_FILE.exists():
        with open(SEGMENTS_FILE) as f:
            d = json.load(f)
        print(
            f"[skip] {SEGMENTS_FILE.name} already present "
            f"(rare={len(d['rare_classes'])} common={len(d['common_classes'])} "
            f"frequent={len(d['frequent_classes'])})"
        )
        return

    print("Computing segments from WILDS train counts…")
    cached = load_cached_features()
    train_labels = cached["train_labels"].numpy()

    counts = [int((train_labels == c).sum()) for c in range(NUM_CLASSES)]
    rare = sorted(c for c in range(NUM_CLASSES) if counts[c] < RARE_THRESHOLD)
    common = sorted(
        c for c in range(NUM_CLASSES) if RARE_THRESHOLD <= counts[c] <= COMMON_THRESHOLD
    )
    frequent = sorted(c for c in range(NUM_CLASSES) if counts[c] > COMMON_THRESHOLD)

    eligible_per_split = {}
    qualifying_per_split = {}
    for split in EVAL_SPLITS:
        labels_s = cached[f"{split}_labels"].numpy()
        meta_s = cached[f"{split}_metadata"].numpy()
        elig = sorted(
            c for c in range(NUM_CLASSES) if counts[c] > 0 and (labels_s == c).sum() > 0
        )
        eligible_per_split[split] = elig

        cam_counts = Counter(meta_s[:, 0])
        qualifying_per_split[split] = sorted(
            int(cam) for cam, n in cam_counts.items() if n >= MIN_CAM_IMAGES
        )

    out = {
        "rare_threshold": RARE_THRESHOLD,
        "common_threshold": COMMON_THRESHOLD,
        "num_classes": NUM_CLASSES,
        "train_class_counts": counts,
        "rare_classes": rare,
        "common_classes": common,
        "frequent_classes": frequent,
        "eligible_per_split": eligible_per_split,
        "qualifying_cameras_per_split": qualifying_per_split,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGMENTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[save] {SEGMENTS_FILE} "
        f"(rare={len(rare)} common={len(common)} frequent={len(frequent)})"
    )


def main() -> None:
    extract_frozen_features()
    build_segments_cache()


if __name__ == "__main__":
    main()
