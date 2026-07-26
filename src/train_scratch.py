"""
End-to-end training for the from-scratch ResNet50 baseline.

Uses a strong training recipe inspired by "ResNet Strikes Back" (Wightman et al., 2021)
with timm's augmentation pipeline:
  - RandAugment (rand-m7-mstd0.5-inc1)
  - Mixup (alpha=0.2) + CutMix (alpha=1.0) with label smoothing (0.1)
  - Random Erasing (prob=0.25, mode=pixel)
  - Linear warmup (5 epochs) + cosine annealing
  - AdamW optimizer

Training strategy (2-phase):
  1. PROBE: run a small grid of LRs for --probe_epochs each, pick the best.
  2. TRAIN: full training with the winning LR for --max_epochs.

This avoids the cost of sweeping 13 LRs for the full epoch budget.

Usage:
    uv run train_scratch.py \
        --config configs/model/cycle_1_thesis/resnet50_scratch.yaml \
        --wandb_project iwildcam_2022_sweep

    # Skip probe and train directly with a known LR:
    uv run train_scratch.py \
        --config configs/model/cycle_1_thesis/resnet50_scratch.yaml \
        --lr 1e-3 --skip_probe
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import timm
import timm.data
import timm.loss
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader

from data.iwildcam import iWildCamDataModule

# Augmentation defaults (A3-ish recipe, adapted for smaller datasets)
AUG_DEFAULTS = dict(
    # RandAugment config string: magnitude 7, 2 ops, magnitude std 0.5,
    # use increasing severity transforms
    rand_augment="rand-m7-mstd0.5-inc1",
    # Mixup / CutMix
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    mixup_prob=1.0,
    mixup_switch_prob=0.5,
    label_smoothing=0.1,
    # Random Erasing
    random_erasing_prob=0.25,
    random_erasing_mode="pixel",
    random_erasing_count=1,
    # Image
    img_size=224,
    interpolation="bicubic",
    color_jitter=0.4,
    # Train
    warmup_epochs=5,
    warmup_lr=1e-6,
)

# Probe grid: a handful of representative LRs
DEFAULT_PROBE_LRS = [5e-5, 5e-4, 2e-3, 5e-3, 2e-2]


def build_train_transform(img_size: int = 224, aug_cfg: dict | None = None):
    """Build training transform using timm's pipeline: RandomResizedCrop +
    hflip + RandAugment + normalize.  Mixup/CutMix and Random Erasing are
    applied separately on GPU batches."""
    cfg = {**AUG_DEFAULTS, **(aug_cfg or {})}
    return timm.data.create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment=cfg["rand_augment"],
        interpolation=cfg["interpolation"],
        color_jitter=cfg["color_jitter"],
        # Random erasing is handled in the model's training_step for
        # consistency with mixup (both operate on GPU tensors).
        re_prob=0.0,
        mean=timm.data.IMAGENET_DEFAULT_MEAN,
        std=timm.data.IMAGENET_DEFAULT_STD,
    )


def build_val_transform(img_size: int = 224):
    return timm.data.create_transform(
        input_size=img_size,
        is_training=False,
        interpolation="bicubic",
        mean=timm.data.IMAGENET_DEFAULT_MEAN,
        std=timm.data.IMAGENET_DEFAULT_STD,
    )


class ScratchResNet50(pl.LightningModule):
    """
    ResNet50 trained end-to-end from random init with a strong aug recipe.

    Mixup/CutMix and Random Erasing are applied inside training_step (on GPU)
    rather than in the dataloader for efficiency and because they need batch-
    level operations.

    Head: BN1d -> Linear (same as linear eval protocol for fair comparison).
    """

    def __init__(
        self,
        num_classes: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        batch_size: int,
        warmup_epochs: int = 5,
        warmup_lr: float = 1e-6,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 1.0,
        mixup_switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
        random_erasing_prob: float = 0.25,
        random_erasing_mode: str = "pixel",
        random_erasing_count: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone: timm resnet50, no pretrained weights
        backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
        feature_dim = backbone.num_features  # 2048
        self.backbone = backbone

        # Classifier head matches the linear-eval protocol
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

        self.num_classes = num_classes

        # Mixup/CutMix and Random Erasing are applied on GPU in training_step
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

        self.random_erasing = RandomErasing(
            probability=random_erasing_prob,
            mode=random_erasing_mode,
            max_count=random_erasing_count,
        )

        # Validation: standard CE with label smoothing
        self.criterion_val = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Training: soft-target CE (handles mixed one-hot targets from mixup)
        self.criterion_train = timm.loss.SoftTargetCrossEntropy()

        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = torchmetrics.classification.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = torchmetrics.classification.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, average=None
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Apply mixup/cutmix on GPU — targets become soft (N, num_classes)
        x, y_mixed = self.mixup_fn(x, y)
        # Apply random erasing after mixup
        x = self.random_erasing(x)

        logits = self(x)
        loss = self.criterion_train(logits, y_mixed)

        # Approximate accuracy using argmax of soft targets
        acc = (logits.argmax(dim=-1) == y_mixed.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion_val(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        preds = logits.argmax(dim=1)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_per_class_acc(preds, y)
        return loss

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        per_class_acc = self.val_per_class_acc.compute()

        self.log("val_f1_macro", f1, prog_bar=True)
        self.log("val_precision_macro", precision, prog_bar=True)
        self.log("val_recall_macro", recall, prog_bar=True)

        for i in range(self.num_classes):
            self.log(f"val_acc_class_{i}", per_class_acc[i])

        self.log("val_acc_std", torch.std(per_class_acc), prog_bar=True)
        self.log("val_acc_min", torch.min(per_class_acc), prog_bar=True)

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_per_class_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        # Linear warmup + cosine annealing
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_lr / self.hparams.lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=0,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        return [optimizer], [scheduler]


def _build_trainer(args, max_epochs, wandb_logger, checkpoint_callback):
    return pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        devices=args.devices,
        accelerator=args.accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        check_val_every_n_epoch=1,
        deterministic=args.deterministic,
        logger=wandb_logger,
        enable_progress_bar=True,
    )


def run_single(
    lr: float,
    model_name: str,
    num_classes: int,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    aug_cfg: dict,
    tag: str = "full",
) -> float:
    """Train a single run. `tag` is 'probe' or 'full'."""
    model = ScratchResNet50(
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        warmup_epochs=aug_cfg.get("warmup_epochs", AUG_DEFAULTS["warmup_epochs"]),
        warmup_lr=aug_cfg.get("warmup_lr", AUG_DEFAULTS["warmup_lr"]),
        mixup_alpha=aug_cfg.get("mixup_alpha", AUG_DEFAULTS["mixup_alpha"]),
        cutmix_alpha=aug_cfg.get("cutmix_alpha", AUG_DEFAULTS["cutmix_alpha"]),
        mixup_prob=aug_cfg.get("mixup_prob", AUG_DEFAULTS["mixup_prob"]),
        mixup_switch_prob=aug_cfg.get(
            "mixup_switch_prob", AUG_DEFAULTS["mixup_switch_prob"]
        ),
        label_smoothing=aug_cfg.get("label_smoothing", AUG_DEFAULTS["label_smoothing"]),
        random_erasing_prob=aug_cfg.get(
            "random_erasing_prob", AUG_DEFAULTS["random_erasing_prob"]
        ),
        random_erasing_mode=aug_cfg.get(
            "random_erasing_mode", AUG_DEFAULTS["random_erasing_mode"]
        ),
        random_erasing_count=aug_cfg.get(
            "random_erasing_count", AUG_DEFAULTS["random_erasing_count"]
        ),
    )

    wandb_logger = pl_loggers.WandbLogger(
        project=args.wandb_project,
        name=f"{model_name}_{tag}_lr{lr:.0e}",
        save_dir="logs/",
        group=f"sweep_{model_name}",
        tags=["scratch", tag, f"lr_{lr:.0e}"],
    )

    ckpt_dir = os.path.join(
        args.checkpoint_dir, f"sweep_{model_name}", f"{tag}_lr_{lr:.0e}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_f1_macro:.4f}",
        save_top_k=1,
        save_weights_only=True,
        mode="max",
        monitor="val_f1_macro",
        auto_insert_metric_name=False,
    )

    trainer = _build_trainer(args, max_epochs, wandb_logger, checkpoint_callback)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_score = checkpoint_callback.best_model_score
    best_score = 0.0 if best_score is None else float(best_score)

    wandb_logger.experiment.finish(quiet=True)
    return best_score


def main():
    parser = ArgumentParser(
        description="Strong-baseline from-scratch ResNet50 training"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--deterministic", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/sweep")
    parser.add_argument("--wandb_project", default="iwildcam_2022_sweep")
    # Phase control
    parser.add_argument(
        "--probe_epochs", default=10, type=int, help="Epochs per LR during probe phase"
    )
    parser.add_argument(
        "--skip_probe",
        action="store_true",
        help="Skip probe phase and train directly with --lr",
    )
    parser.add_argument(
        "--lr", default=None, type=float, help="Override LR (required if --skip_probe)"
    )
    args = parser.parse_args()

    if args.skip_probe and args.lr is None:
        parser.error("--lr is required when --skip_probe is set")

    try:
        with open(args.config) as f:
            train_config = yaml.safe_load(f)
        with open(train_config["dataset"]) as f:
            data_config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    aug_cfg = {**AUG_DEFAULTS, **train_config.get("augmentation", {})}

    base_dir = Path(data_config["paths"]["base_dir"])
    train_det = data_config["paths"]["train"].get("detections")
    val_det = data_config["paths"].get("val", {}).get("detections")
    data_module = iWildCamDataModule(
        root_dir=base_dir,
        metadata_path=Path(data_config["paths"]["train"]["metadata"]),
        image_dir=Path(data_config["paths"]["train"]["image_dir"]),
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["dataloader"]["num_workers"],
        train_transform=build_train_transform(aug_cfg["img_size"], aug_cfg),
        val_transform=build_val_transform(aug_cfg["img_size"]),
        train_detections_path=base_dir / train_det if train_det else None,
        val_detections_path=base_dir / val_det if val_det else None,
    )
    data_module.setup("fit")

    model_name = train_config["model"]["name"]
    batch_size = train_config["training"]["batch_size"]
    max_epochs = train_config["training"].get("max_epochs", 50)
    weight_decay = train_config["training"]["optimizer"].get("weight_decay", 1e-4)
    num_classes = data_module.num_classes

    _tl = data_module.train_dataloader()
    train_loader = DataLoader(
        _tl.dataset,
        batch_size=_tl.batch_size,
        shuffle=True,
        num_workers=_tl.num_workers,
        pin_memory=_tl.pin_memory,
        drop_last=True,
    )
    val_loader = data_module.val_dataloader()

    if not args.skip_probe:
        probe_lrs = train_config.get("probe_lrs", DEFAULT_PROBE_LRS)
        probe_epochs = train_config.get("probe_epochs", args.probe_epochs)

        print(f"\n{'=' * 60}")
        print(f"PHASE 1: PROBE — {len(probe_lrs)} LRs x {probe_epochs} epochs")
        print(f"{'=' * 60}")

        probe_results = {}
        for lr in probe_lrs:
            pl.seed_everything(args.seed, workers=True)
            score = run_single(
                lr=lr,
                model_name=model_name,
                num_classes=num_classes,
                weight_decay=weight_decay,
                max_epochs=probe_epochs,
                batch_size=batch_size,
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                aug_cfg=aug_cfg,
                tag="probe",
            )
            probe_results[lr] = score
            print(f"  lr={lr:.0e}  =>  val_f1={score:.4f}")

        best_lr = max(probe_results, key=probe_results.get)
        print(f"\nProbe winner: lr={best_lr:.0e} (val_f1={probe_results[best_lr]:.4f})")
        print(f"{'=' * 60}\n")
    else:
        best_lr = args.lr
        print(f"\nSkipping probe, using lr={best_lr:.0e}")

    print(f"\n{'=' * 60}")
    print(f"PHASE 2: FULL TRAINING — lr={best_lr:.0e}, {max_epochs} epochs")
    print(f"{'=' * 60}")

    pl.seed_everything(args.seed, workers=True)
    final_score = run_single(
        lr=best_lr,
        model_name=model_name,
        num_classes=num_classes,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        aug_cfg=aug_cfg,
        tag="full",
    )
    print(f"\nFinal result: lr={best_lr:.0e}, val_f1={final_score:.4f}")

    results_path = os.path.join(
        args.checkpoint_dir, f"sweep_{model_name}", "results.yaml"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_data = {
        "model": model_name,
        "recipe": "resnet_strikes_back_a3_adapted",
        "best_lr": float(best_lr),
        "final_val_f1": float(final_score),
        "max_epochs": max_epochs,
        "augmentation": {k: str(v) for k, v in aug_cfg.items()},
    }
    if not args.skip_probe:
        results_data["probe_results"] = {
            f"{lr:.0e}": float(s) for lr, s in probe_results.items()
        }
    with open(results_path, "w") as f:
        yaml.dump(results_data, f)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
