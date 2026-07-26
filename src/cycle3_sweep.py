"""
Cycle 3: Classification strategy sweep on frozen DINOv2 features.

Tests whether the classifier (not the backbone) is the bottleneck for
rare-class performance.  Five strategies are evaluated on cached features:

  ce         — Linear + cross-entropy baseline
  crt        — Linear + CE with class-balanced sampling (Kang et al., 2020)
  ncm        — Nearest-class-mean, cosine similarity (no training)
  mlp_ce     — Single-hidden-layer MLP + CE (Chen et al., 2019)
  mlp_focal  — Single-hidden-layer MLP + focal loss (Lin et al., 2017)

Model selection uses val macro-F1 (not accuracy) to favor rare classes.

Usage (from project root):
    uv run src/cycle3_sweep.py \
        --config src/configs/model/cycle_3_thesis/dinov2_vitb14.yaml \
        --strategy ce --wandb_project iwildcam2022_cycle3_sweep \
        --checkpoint_dir checkpoints/cycle3_sweep

    # Run all strategies sequentially:
    uv run src/cycle3_sweep.py ... --strategy all
"""

import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data.iwildcam import iWildCamDataModule
from data.serengeti import SerengetiDataModule
from models.base_classifier import BaseClassifier

DEFAULT_LR_GRID = [
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
    0.1,
]

FOCAL_GAMMAS = [0.5, 1.0, 2.0, 5.0]

MLP_HIDDEN_DIM = 256


def focal_loss(logits, labels, gamma=2.0):
    """Focal loss (Lin et al., 2017)."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def build_mlp_head(feat_dim, num_classes, hidden_dim=MLP_HIDDEN_DIM):
    """
    Single-hidden-layer MLP classifier (Chen et al., 2019).

    Minimal non-linear extension of the linear probe: BN → Linear → ReLU →
    Linear. Tests whether a non-linear decision boundary improves over the
    linear baseline without adding confounds (dropout, multiple layers).
    BN follows Park et al. (2023) protocol used in the linear baseline.
    """
    return nn.Sequential(
        nn.BatchNorm1d(feat_dim),
        nn.Linear(feat_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def run_ncm(train_features, train_labels, val_features, val_labels, num_classes):
    """
    Nearest-class-mean classifier with cosine similarity.
    No training required — computes class centroids from training features
    and assigns each val image to the nearest centroid.
    """
    centroids = torch.zeros(num_classes, train_features.shape[1])
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            centroids[c] = train_features[mask].mean(dim=0)

    centroids_norm = F.normalize(centroids, dim=1)
    val_norm = F.normalize(val_features, dim=1)
    preds = (val_norm @ centroids_norm.T).argmax(dim=1)

    acc = (preds == val_labels).float().mean().item()
    f1 = torchmetrics.functional.f1_score(
        preds,
        val_labels,
        task="multiclass",
        num_classes=num_classes,
        average="macro",
    ).item()

    return {"val_acc": acc, "val_f1_macro": f1}


def make_balanced_loader(cached_ds, batch_size, num_classes):
    """
    DataLoader with class-balanced sampling (Kang et al., 2020).
    Each class has equal probability of being sampled.
    """
    labels = cached_ds.tensors[1]
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    sample_weights = 1.0 / (counts[labels] + 1e-8)
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
    return DataLoader(
        cached_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )


def build_data_module(data_config, train_config):
    dataset_key = data_config["name"].lower()
    if dataset_key == "iwildcam":
        base_dir = Path(data_config["paths"]["base_dir"])
        train_det = data_config["paths"]["train"].get("detections")
        val_det = data_config["paths"].get("val", {}).get("detections")
        return iWildCamDataModule(
            root_dir=base_dir,
            metadata_path=Path(data_config["paths"]["train"]["metadata"]),
            image_dir=Path(data_config["paths"]["train"]["image_dir"]),
            batch_size=train_config["training"]["batch_size"],
            num_workers=train_config["dataloader"]["num_workers"],
            train_detections_path=base_dir / train_det if train_det else None,
            val_detections_path=base_dir / val_det if val_det else None,
        )
    elif dataset_key == "serengeti":
        return SerengetiDataModule(
            root_dir=Path(data_config["paths"]["base_dir"]),
            batch_size=train_config["training"]["batch_size"],
            num_workers=train_config["dataloader"]["num_workers"],
        )
    raise ValueError(f"Unsupported dataset '{dataset_key}'")


def check_existing_run(checkpoint_dir, group_name, lr):
    """Check if a run already completed. Returns best val_f1 or None."""
    ckpt_dir = os.path.join(checkpoint_dir, group_name, f"lr_{lr:.0e}")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
    best_f1 = 0.0
    for f in ckpts:
        try:
            best_f1 = max(best_f1, float(f.rsplit("-", 1)[1].replace(".ckpt", "")))
        except (ValueError, IndexError):
            continue
    return best_f1


def run_single(
    lr,
    train_loader,
    val_loader,
    args,
    run_name,
    group_name,
    max_epochs,
    num_classes,
    backbone,
    batch_size,
    feat_dim,
    criterion_fn=None,
    use_mlp=False,
    tags=None,
):
    """Train one configuration, return best val_f1_macro."""
    existing = check_existing_run(args.checkpoint_dir, group_name, lr)
    if existing is not None:
        print(f"  [SKIP] {run_name} — already done (val_f1={existing:.4f})")
        return existing

    model = BaseClassifier(
        num_classes=num_classes,
        backbone=backbone,
        frozen=True,
        freeze_strategy="all",
        lr=lr,
        weight_decay=0.0,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    model._cached_mode = True

    if use_mlp:
        model.classifier = build_mlp_head(feat_dim, num_classes)

    if criterion_fn is not None:
        model.criterion = criterion_fn

    # Strip backbone weights from checkpoint (reduces ~330MB to <1MB)
    def _slim_on_save(checkpoint):
        sd = checkpoint.get("state_dict", {})
        for key in list(sd.keys()):
            if key.startswith("backbone."):
                del sd[key]

    model.on_save_checkpoint = _slim_on_save

    wandb_logger = pl_loggers.WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir="logs/",
        group=group_name,
        tags=tags or ["cycle3"],
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, group_name, f"lr_{lr:.0e}"),
        filename="{epoch:02d}-{val_f1_macro:.4f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        mode="max",
        monitor="val_f1_macro",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        devices=args.devices,
        accelerator=args.accelerator,
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        check_val_every_n_epoch=1,
        deterministic=args.deterministic,
        logger=wandb_logger,
        enable_progress_bar=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best = ckpt_cb.best_model_score
    best = float(best) if best is not None else 0.0

    wandb_logger.experiment.finish(quiet=True)
    return best


def main():
    parser = ArgumentParser(description="Cycle 3: classification strategy sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["ce", "crt", "ncm", "mlp_ce", "mlp_focal", "all"],
        help="Classification strategy to sweep",
    )
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--deterministic", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/cycle3_sweep")
    parser.add_argument("--wandb_project", default="iwildcam2022_cycle3_sweep")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            train_config = yaml.safe_load(f)
        with open(train_config["dataset"]) as f:
            data_config = yaml.safe_load(f)
        model_config = train_config["model"]
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    # Features are cached to disk once and reused across runs
    backbone = model_config["backbone"]
    batch_size = train_config["training"]["batch_size"]
    max_epochs = train_config["training"].get("max_epochs", 10)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    cache_path = os.path.join(args.checkpoint_dir, f"cached_features_{backbone}.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        train_features = cached["train_features"]
        train_labels = cached["train_labels"]
        val_features = cached["val_features"]
        val_labels = cached["val_labels"]
        num_classes = cached["num_classes"]
    else:
        data_module = build_data_module(data_config, train_config)
        # Features are cached once and reused across every strategy/LR/epoch,
        # so train-time augmentation would just be a frozen, one-off corruption
        # baked into the cache — use the unaugmented val transform for training too.
        data_module.transform_train = data_module.transform_val
        data_module.setup("fit")
        num_classes = data_module.num_classes

        print(f"Caching features for {backbone}...")
        temp_model = BaseClassifier(
            num_classes=num_classes,
            backbone=backbone,
            frozen=True,
            freeze_strategy="all",
            lr=0.001,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp_model.to(device)

        cached_train_ds = temp_model.precompute_features(data_module.train_dataloader())
        cached_val_ds = temp_model.precompute_features(data_module.val_dataloader())
        del temp_model
        torch.cuda.empty_cache()

        train_features = cached_train_ds.tensors[0]
        train_labels = cached_train_ds.tensors[1]
        val_features = cached_val_ds.tensors[0]
        val_labels = cached_val_ds.tensors[1]

        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "val_features": val_features,
                "val_labels": val_labels,
                "num_classes": num_classes,
            },
            cache_path,
        )
        print(f"Features saved to {cache_path}")

    feat_dim = train_features.shape[1]
    print(f"Cached {len(train_labels)} train + {len(val_labels)} val (dim={feat_dim})")

    cached_train_ds = TensorDataset(train_features, train_labels)
    cached_val_ds = TensorDataset(val_features, val_labels)

    train_loader = DataLoader(
        cached_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        cached_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Shared kwargs for run_single
    common = dict(
        max_epochs=max_epochs,
        num_classes=num_classes,
        backbone=backbone,
        batch_size=batch_size,
        feat_dim=feat_dim,
    )
    lr_grid = train_config.get("lr_grid", DEFAULT_LR_GRID)

    strategies = (
        ["ce", "crt", "ncm", "mlp_ce", "mlp_focal"]
        if args.strategy == "all"
        else [args.strategy]
    )

    all_results = {}

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'=' * 60}")

        if strategy == "ncm":
            result = run_ncm(
                train_features,
                train_labels,
                val_features,
                val_labels,
                num_classes,
            )
            all_results[("ncm",)] = result["val_f1_macro"]
            print(
                f"  NCM  =>  val_acc={result['val_acc']:.4f}  "
                f"val_f1={result['val_f1_macro']:.4f}"
            )

        elif strategy == "ce":
            for lr in lr_grid:
                pl.seed_everything(args.seed, workers=True)
                f1 = run_single(
                    lr=lr,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    args=args,
                    run_name=f"ce_lr{lr:.0e}",
                    group_name="sweep_ce",
                    tags=["cycle3", "ce"],
                    **common,
                )
                all_results[("ce", lr)] = f1
                print(f"  CE  lr={lr:.0e}  =>  val_f1={f1:.4f}")

        elif strategy == "crt":
            balanced_loader = make_balanced_loader(
                cached_train_ds,
                batch_size,
                num_classes,
            )
            for lr in lr_grid:
                pl.seed_everything(args.seed, workers=True)
                f1 = run_single(
                    lr=lr,
                    train_loader=balanced_loader,
                    val_loader=val_loader,
                    args=args,
                    run_name=f"crt_lr{lr:.0e}",
                    group_name="sweep_crt",
                    tags=["cycle3", "crt"],
                    **common,
                )
                all_results[("crt", lr)] = f1
                print(f"  cRT  lr={lr:.0e}  =>  val_f1={f1:.4f}")

        elif strategy == "mlp_ce":
            for lr in lr_grid:
                pl.seed_everything(args.seed, workers=True)
                f1 = run_single(
                    lr=lr,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    args=args,
                    run_name=f"mlp_ce_lr{lr:.0e}",
                    group_name="sweep_mlp_ce",
                    use_mlp=True,
                    tags=["cycle3", "mlp_ce"],
                    **common,
                )
                all_results[("mlp_ce", lr)] = f1
                print(f"  MLP+CE  lr={lr:.0e}  =>  val_f1={f1:.4f}")

        elif strategy == "mlp_focal":
            for gamma in FOCAL_GAMMAS:
                for lr in lr_grid:
                    pl.seed_everything(args.seed, workers=True)
                    criterion = partial(focal_loss, gamma=gamma)
                    f1 = run_single(
                        lr=lr,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        args=args,
                        run_name=f"mlp_focal_g{gamma}_lr{lr:.0e}",
                        group_name=f"sweep_mlp_focal_g{gamma}",
                        use_mlp=True,
                        criterion_fn=criterion,
                        tags=["cycle3", "mlp_focal", f"gamma={gamma}"],
                        **common,
                    )
                    all_results[("mlp_focal", gamma, lr)] = f1
                    print(f"  MLP+Focal g={gamma}  lr={lr:.0e}  =>  val_f1={f1:.4f}")

    print(f"\n{'=' * 60}")
    print("CYCLE 3 RESULTS")
    print(f"{'=' * 60}")

    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    best_key, best_f1 = sorted_results[0]

    for key, f1 in sorted_results:
        marker = " <-- best" if f1 == best_f1 else ""
        print(f"  {key}  =>  val_f1={f1:.4f}{marker}")

    print(f"\nBest: {best_key} with val_f1={best_f1:.4f}")

    results_path = os.path.join(
        args.checkpoint_dir,
        f"results_{args.strategy}.yaml",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    serializable = {str(k): float(v) for k, v in all_results.items()}
    with open(results_path, "w") as f:
        yaml.dump(
            {
                "backbone": backbone,
                "strategy": args.strategy,
                "best_config": str(best_key),
                "best_val_f1": float(best_f1),
                "all_results": serializable,
            },
            f,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
