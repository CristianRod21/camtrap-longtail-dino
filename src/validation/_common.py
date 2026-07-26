"""Shared utilities for the WILDS validation scripts.

Reuses the metric / bootstrap / head-training conventions established in
cycles 3-4 (frozen-feature linear probing, cRT balanced sampling, cluster
bootstrap by camera) so this validation pass is directly comparable to the
cycle results. All scripts in this directory import from here.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Make src/ importable so scripts can use BaseClassifier and the data utils
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "validation"
RESULTS_DIR = CHECKPOINT_DIR / "results"

CACHE_FILE = Path("/path/to/wilds_cache/cached_features_wilds.pt")
SEGMENTS_FILE = RESULTS_DIR / "wilds_class_segments.json"

RARE_THRESHOLD = 10
COMMON_THRESHOLD = 100
MIN_CAM_IMAGES = 15
N_BOOTSTRAP = 2000
SEED = 42
FEAT_DIM = 768
BACKBONE = "dinov2_vitb14"
BLANK_CLASS = 0
NUM_CLASSES = 182

# Same 13-LR grid used in cycles 1/3
LR_GRID = [
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
]

EVAL_SPLITS = ["id_val", "id_test", "val", "test"]
SPLIT_LABEL = {
    "id_val": "ID val (seen locations)",
    "id_test": "ID test (seen locations)",
    "val": "OOD val (unseen locations)",
    "test": "OOD test (unseen locations)",
}


def load_segments() -> dict:
    with open(SEGMENTS_FILE) as f:
        return json.load(f)


def load_cached_features():
    return torch.load(CACHE_FILE, map_location="cpu", weights_only=True)


def get_wilds_dataset():
    """Return a WILDS iWildCam dataset object (for `dataset.eval(...)`)."""
    from data.wilds import get_dataset

    return get_dataset()


def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    eligible_classes: set,
    rare_classes: set,
    common_classes: set,
    frequent_classes: set,
) -> dict:
    """Macro / Blank / Species / Head / Medium / Tail F1.

    All F1s are restricted to the per-split eligible class set (classes with
    >0 train samples AND >0 split samples). Blank/Species/segments are
    further intersected with the eligible set.
    """
    label_list = sorted(eligible_classes)
    label_to_idx = {c: i for i, c in enumerate(label_list)}
    per_class = f1_score(
        targets, preds, labels=label_list, average=None, zero_division=0
    )

    res: dict = {}
    res["Macro-F1"] = f1_score(
        targets, preds, labels=label_list, average="macro", zero_division=0
    )

    # Blank: per-class F1 for class 0, only if it's eligible in this split
    if BLANK_CLASS in label_to_idx:
        res["Blank-F1"] = float(per_class[label_to_idx[BLANK_CLASS]])
    else:
        res["Blank-F1"] = float("nan")

    # Species: macro-F1 over eligible_classes \ {0}, computed only on rows
    # whose target is in that set
    species_classes = eligible_classes - {BLANK_CLASS}
    sp_labels = sorted(species_classes)
    sp_mask = np.isin(targets, list(species_classes))
    res["Species-F1"] = (
        f1_score(
            targets[sp_mask],
            preds[sp_mask],
            labels=sp_labels,
            average="macro",
            zero_division=0,
        )
        if sp_mask.sum() > 0
        else 0.0
    )

    # Per-segment F1 — restricted to eligible classes within each segment
    for seg_name, seg_global in (
        ("Head", frequent_classes),
        ("Medium", common_classes),
        ("Tail", rare_classes),
    ):
        seg_eligible = seg_global & eligible_classes
        seg_labels = sorted(seg_eligible)
        seg_mask = np.isin(targets, list(seg_eligible))
        res[f"{seg_name}-F1"] = (
            f1_score(
                targets[seg_mask],
                preds[seg_mask],
                labels=seg_labels,
                average="macro",
                zero_division=0,
            )
            if seg_mask.sum() > 0
            else 0.0
        )

    return res


def dataset_eval_macro_f1(
    dataset, preds: torch.Tensor, labels: torch.Tensor, metadata: torch.Tensor
) -> float:
    """Run the WILDS official `dataset.eval(...)` and return macro-F1.

    Secondary track for leaderboard-parity reporting alongside the
    eligible-class metric above (see `03_k4_crt.py`/`04_evaluate_and_report.py`).
    """
    metrics, _ = dataset.eval(preds, labels, metadata)
    if "F1-macro_all" in metrics:
        return float(metrics["F1-macro_all"])
    for k, v in metrics.items():
        if "f1" in k.lower() and "macro" in k.lower():
            return float(v)
    raise KeyError(f"Could not find macro-F1 in WILDS eval metrics: {list(metrics)}")


def holm_bonferroni(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni correction. Returns list of (adjusted_p, significant)."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [None] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = (cummax, cummax <= alpha)
    return adjusted


def cluster_bootstrap_f1(
    targets: np.ndarray,
    preds: np.ndarray,
    locations: np.ndarray,
    eligible_classes: set,
    n_boot: int = N_BOOTSTRAP,
    groups: dict | None = None,
    rng: np.random.Generator | None = None,
):
    """Per-condition cluster bootstrap on cameras.

    Returns:
        point: float — macro-F1 over eligible classes
        ci:    np.ndarray of shape (2,) — 95% CI [low, high]
        samples: np.ndarray of shape (n_boot,) — bootstrap distribution
        groups_out: dict — {group_name: (point, ci, samples)} per group
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    label_list = sorted(eligible_classes)

    cameras = np.unique(locations)
    cam_idx = {cam: np.where(locations == cam)[0] for cam in cameras}

    boot_f1 = []
    boot_groups = defaultdict(list)
    for _ in range(n_boot):
        sampled_cams = rng.choice(cameras, size=len(cameras), replace=True)
        idx = np.concatenate([cam_idx[c] for c in sampled_cams])
        t_b, p_b = targets[idx], preds[idx]
        boot_f1.append(
            f1_score(t_b, p_b, labels=label_list, average="macro", zero_division=0)
        )
        if groups:
            for gname, gcls in groups.items():
                gm = np.isin(t_b, list(gcls))
                if gm.sum() > 0:
                    boot_groups[gname].append(
                        f1_score(
                            t_b[gm],
                            p_b[gm],
                            labels=sorted(gcls),
                            average="macro",
                            zero_division=0,
                        )
                    )
                else:
                    boot_groups[gname].append(0.0)

    boot_f1 = np.array(boot_f1)
    point = f1_score(
        targets, preds, labels=label_list, average="macro", zero_division=0
    )
    ci = np.percentile(boot_f1, [2.5, 97.5])

    group_out = {}
    if groups:
        for gname, vals in boot_groups.items():
            vals = np.array(vals)
            gmask = np.isin(targets, list(groups[gname]))
            if gmask.sum() > 0:
                gpt = f1_score(
                    targets[gmask],
                    preds[gmask],
                    labels=sorted(groups[gname]),
                    average="macro",
                    zero_division=0,
                )
            else:
                gpt = 0.0
            group_out[gname] = (gpt, np.percentile(vals, [2.5, 97.5]), vals)

    return float(point), ci, boot_f1, group_out


def paired_cluster_bootstrap(
    targets: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    locations: np.ndarray,
    eligible_classes: set,
    n_boot: int = N_BOOTSTRAP,
    groups: dict | None = None,
    rng: np.random.Generator | None = None,
):
    """Paired bootstrap: gap = F1(a) - F1(b) per resample, paired by camera.

    Returns:
        gap_overall: dict with point, ci, samples, p_raw
        gap_by_group: dict {gname: (point, ci, samples, p_raw)}
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    label_list = sorted(eligible_classes)

    cameras = np.unique(locations)
    cam_idx = {cam: np.where(locations == cam)[0] for cam in cameras}

    gap_samples = []
    gap_group_samples = defaultdict(list)
    for _ in range(n_boot):
        sampled_cams = rng.choice(cameras, size=len(cameras), replace=True)
        idx = np.concatenate([cam_idx[c] for c in sampled_cams])
        t = targets[idx]
        f_a = f1_score(
            t, preds_a[idx], labels=label_list, average="macro", zero_division=0
        )
        f_b = f1_score(
            t, preds_b[idx], labels=label_list, average="macro", zero_division=0
        )
        gap_samples.append(f_a - f_b)
        if groups:
            for gname, gcls in groups.items():
                gm = np.isin(t, list(gcls))
                if gm.sum() > 0:
                    fa = f1_score(
                        t[gm],
                        preds_a[idx][gm],
                        labels=sorted(gcls),
                        average="macro",
                        zero_division=0,
                    )
                    fb = f1_score(
                        t[gm],
                        preds_b[idx][gm],
                        labels=sorted(gcls),
                        average="macro",
                        zero_division=0,
                    )
                    gap_group_samples[gname].append(fa - fb)
                else:
                    gap_group_samples[gname].append(0.0)

    gap_samples = np.array(gap_samples)
    point = f1_score(
        targets, preds_a, labels=label_list, average="macro", zero_division=0
    ) - f1_score(targets, preds_b, labels=label_list, average="macro", zero_division=0)
    ci = np.percentile(gap_samples, [2.5, 97.5])
    p_raw = 2 * min((gap_samples <= 0).mean(), (gap_samples >= 0).mean())

    gap_overall = dict(
        point=float(point), ci=ci, samples=gap_samples, p_raw=float(p_raw)
    )

    gap_by_group = {}
    if groups:
        for gname, vals in gap_group_samples.items():
            vals = np.array(vals)
            gmask = np.isin(targets, list(groups[gname]))
            if gmask.sum() > 0:
                pt = f1_score(
                    targets[gmask],
                    preds_a[gmask],
                    labels=sorted(groups[gname]),
                    average="macro",
                    zero_division=0,
                ) - f1_score(
                    targets[gmask],
                    preds_b[gmask],
                    labels=sorted(groups[gname]),
                    average="macro",
                    zero_division=0,
                )
            else:
                pt = 0.0
            ci_g = np.percentile(vals, [2.5, 97.5])
            p_g = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
            gap_by_group[gname] = dict(
                point=float(pt), ci=ci_g, samples=vals, p_raw=float(p_g)
            )

    return gap_overall, gap_by_group


def make_head(feat_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, num_classes))


def make_balanced_sampler(
    labels: torch.Tensor, num_classes: int
) -> WeightedRandomSampler:
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    weights = 1.0 / (counts[labels] + 1e-8)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


@torch.no_grad()
def eval_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    eligible_classes: set,
    device: torch.device,
    batch: int = 512,
):
    head.eval()
    preds_chunks = []
    for i in range(0, len(features), batch):
        x = features[i : i + batch].to(device)
        logits = head(x).cpu()
        preds_chunks.append(logits.argmax(1))
    preds = torch.cat(preds_chunks).numpy()
    label_list = sorted(eligible_classes)
    f1 = f1_score(
        labels.numpy(),
        preds,
        labels=label_list,
        average="macro",
        zero_division=0,
    )
    return preds, float(f1)


def train_head_on_features(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_eligible_classes: set,
    num_classes: int,
    lr: float,
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 10,
    balanced: bool = False,
    seed: int = SEED,
):
    """Train BN+Linear head on cached features. SGD-momentum-cosine, lr*batch/256.

    Returns (best_state, best_val_f1).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = make_head(train_features.shape[1], num_classes).to(device)
    scaled_lr = lr * batch_size / 256.0
    opt = torch.optim.SGD(
        head.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

    ds = TensorDataset(train_features, train_labels)
    if balanced:
        sampler = make_balanced_sampler(train_labels, num_classes)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    best_state = None
    best_val_f1 = -1.0
    for ep in range(epochs):
        head.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            opt.step()
        sched.step()
        _, val_f1 = eval_head(
            head, val_features, val_labels, val_eligible_classes, device
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
    return best_state, best_val_f1


def build_partial_unfreeze_model(num_classes: int, unfreeze_k: int = 4):
    """Mirror of build_model() in src/cycle4_partial_unfreeze.py.

    unfreeze_k=0 gives an all-frozen backbone (used by 00_preflight.py).
    """
    from models.base_classifier import BaseClassifier

    model = BaseClassifier(
        num_classes=num_classes,
        backbone=BACKBONE,
        frozen=True,
        freeze_strategy="all",
        lr=1e-3,
        max_epochs=5,
        batch_size=32,
    )
    bb = model.backbone
    n_blocks = len(bb.blocks)
    for i, block in enumerate(bb.blocks):
        trainable = i >= n_blocks - unfreeze_k
        for p in block.parameters():
            p.requires_grad = trainable
    for p in bb.norm.parameters():
        p.requires_grad = True
    return model, n_blocks
