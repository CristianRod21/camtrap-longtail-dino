"""
Linear evaluation LR sweep.

Follows the DINOv2 evaluation protocol (Oquab et al., 2023): sweep over a
grid of learning rates and report the best result per backbone. This ensures
fair comparison across backbones with different feature distributions.

The DINOv2 eval code uses:
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
                      1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]

Usage:
    uv run src/linear_eval_sweep.py --config src/configs/model/dinov2_vitb14.yaml \
        --wandb_project iwildcam_2022_sweep
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from data.iwildcam import iWildCamDataModule
from data.serengeti import SerengetiDataModule
from models.base_classifier import BaseClassifier

# DINOv2 standard LR grid (from dinov2/eval/linear.py)
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
    else:
        raise ValueError(f"Unsupported dataset '{dataset_key}'")


def run_single_lr(
    lr: float,
    model_config: dict,
    train_config: dict,
    cached_train_loader: DataLoader,
    cached_val_loader: DataLoader,
    args,
    backbone: str,
    model_name: str,
    max_epochs: int,
    num_classes: int,
) -> float:
    """Train a single LR and return the best val_f1_macro."""

    model = BaseClassifier(
        num_classes=num_classes,
        backbone=backbone,
        frozen=True,
        freeze_strategy="all",
        lr=lr,
        weight_decay=0.0,
        max_epochs=max_epochs,
        batch_size=train_config["training"]["batch_size"],
    )
    model._cached_mode = True

    wandb_logger = pl_loggers.WandbLogger(
        project=args.wandb_project,
        name=f"{model_name}_lr{lr:.0e}",
        save_dir="logs/",
        group=f"sweep_{model_name}",
        tags=["lr_sweep"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            args.checkpoint_dir, f"sweep_{model_name}", f"lr_{lr:.0e}"
        ),
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
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        check_val_every_n_epoch=1,
        deterministic=args.deterministic,
        logger=wandb_logger,
        enable_progress_bar=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=cached_train_loader,
        val_dataloaders=cached_val_loader,
    )

    best_acc = checkpoint_callback.best_model_score
    if best_acc is None:
        best_acc = 0.0
    else:
        best_acc = float(best_acc)

    wandb_logger.experiment.finish(quiet=True)
    return best_acc


def main():
    parser = ArgumentParser(description="LR sweep for linear evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config"
    )
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--deterministic", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/sweep")
    parser.add_argument("--wandb_project", default="iwildcam_2022_sweep")
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

    data_module = build_data_module(data_config, train_config)
    data_module.num_workers = 0
    # Features are cached once and reused across all epochs/LRs, so augmentation
    # would just be frozen noise — use the unaugmented val transform for training too.
    data_module.transform_train = data_module.transform_val
    data_module.setup("fit")
    # Features are cached once here and reused across every LR in the sweep
    model_name = model_config.get("name", "unnamed")
    backbone = model_config["backbone"]
    batch_size = train_config["training"]["batch_size"]
    num_workers = train_config["dataloader"]["num_workers"]  # noqa: F841
    max_epochs = train_config["training"].get("max_epochs", 10)
    num_classes = data_module.num_classes

    print(f"Caching features for {backbone}...")
    # Build a temporary model just for feature extraction
    temp_model = BaseClassifier(
        num_classes=num_classes,
        backbone=backbone,
        frozen=True,
        freeze_strategy="all",
        lr=0.001,  # doesn't matter, just for init
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model.to(device)

    cached_train_ds = temp_model.precompute_features(data_module.train_dataloader())
    cached_val_ds = temp_model.precompute_features(data_module.val_dataloader())

    del temp_model
    torch.cuda.empty_cache()

    feat_dim = cached_train_ds.tensors[0].shape[1]
    n_train = len(cached_train_ds)
    n_val = len(cached_val_ds)
    print(f"Cached {n_train} train + {n_val} val (dim={feat_dim})")

    cached_train_loader = DataLoader(
        cached_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    cached_val_loader = DataLoader(
        cached_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    lr_grid = train_config.get("lr_grid", DEFAULT_LR_GRID)

    results = {}
    print(f"\nSweeping {len(lr_grid)} LRs for {backbone} ({max_epochs} epochs each)")
    print("=" * 60)

    for lr in lr_grid:
        pl.seed_everything(args.seed, workers=True)

        acc = run_single_lr(
            lr=lr,
            model_config=model_config,
            train_config=train_config,
            cached_train_loader=cached_train_loader,
            cached_val_loader=cached_val_loader,
            args=args,
            backbone=backbone,
            model_name=model_name,
            max_epochs=max_epochs,
            num_classes=num_classes,
        )
        results[lr] = acc
        print(f"  lr={lr:.0e}  =>  val_f1_macro={acc:.4f}")

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    for lr, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- best" if acc == max(results.values()) else ""
        print(f"  lr={lr:.0e}  =>  val_f1_macro={acc:.4f}{marker}")

    best_lr = max(results, key=results.get)
    best_acc = results[best_lr]
    print(f"\nBest: lr={best_lr:.0e} with val_f1_macro={best_acc:.4f}")

    # Save results
    results_path = os.path.join(
        args.checkpoint_dir, f"sweep_{model_name}", "results.yaml"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(
            {
                "backbone": backbone,
                "best_lr": float(best_lr),
                "best_val_f1_macro": float(best_acc),
                "all_results": {f"{lr:.0e}": float(acc) for lr, acc in results.items()},
            },
            f,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
