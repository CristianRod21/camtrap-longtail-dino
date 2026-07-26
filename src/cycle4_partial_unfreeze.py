"""
Cycle 4: Partial fine-tuning of DINOv2 — backbone LR sweep at fixed unfreeze depth.

Tests whether unfreezing the last K transformer blocks (K in {2, 4}) and
training end-to-end with plain CE moves the unseen-camera macro-F1.
Cycle 3 established that no classifier-level intervention on frozen features
breaks the seen/unseen gap; this sweep tests the corollary at the
representation level.

Sweep over backbone learning rates: [5e-6, 1e-5, 2e-5, 5e-5, 1e-4].
Head LR is fixed at 1e-3 (matches partial_unfreeze.ipynb).

Usage (from project root):
    uv run src/cycle4_partial_unfreeze.py --unfreeze_k 2 \
        --wandb_project iwildcam2022_cycle4_sweep \
        --checkpoint_dir checkpoints/cycle4_sweep

    uv run src/cycle4_partial_unfreeze.py --unfreeze_k 4 ...
"""

import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score

from data.iwildcam import iWildCamDataModule
from models.base_classifier import BaseClassifier

BACKBONE_LR_GRID = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

HEAD_LR = 1e-3
BACKBONE = "dinov2_vitb14"
BATCH_SIZE = 32
EPOCHS = 5
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 500
USE_AMP = True
SEED = 42

ROOT = Path("/path/to/iwildcam_224/")
META = Path("metadata/metadata/iwildcam2022_train_annotations.json")


def build_model(num_classes, unfreeze_k):
    model = BaseClassifier(
        num_classes=num_classes,
        backbone=BACKBONE,
        frozen=True,
        freeze_strategy="all",
        lr=HEAD_LR,
        max_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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


def param_groups(model, backbone_lr):
    backbone_trainable = [
        p for _, p in model.backbone.named_parameters() if p.requires_grad
    ]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return (
        [
            {
                "params": backbone_trainable,
                "lr": backbone_lr,
                "weight_decay": WEIGHT_DECAY,
            },
            {"params": head_params, "lr": HEAD_LR, "weight_decay": 0.0},
        ],
        sum(p.numel() for p in backbone_trainable),
        sum(p.numel() for p in head_params),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets, f1_score(targets, preds, average="macro", zero_division=0)


def run_dir(checkpoint_dir, unfreeze_k, backbone_lr):
    return Path(checkpoint_dir) / f"k{unfreeze_k}_lr{backbone_lr:.0e}"


def existing_checkpoint(run_dir_path):
    ckpt = run_dir_path / "best.ckpt"
    if not ckpt.exists():
        return None
    try:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        return meta
    except Exception:
        return None


def run_single(
    unfreeze_k,
    backbone_lr,
    dm,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    args,
    device,
):
    out = run_dir(args.checkpoint_dir, unfreeze_k, backbone_lr)
    out.mkdir(parents=True, exist_ok=True)

    existing = existing_checkpoint(out)
    if existing is not None:
        best_val_f1 = existing.get("best_val_f1", float("nan"))
        test_f1 = existing.get("test_f1_macro", None)
        print(
            f"  [SKIP] k={unfreeze_k} bb_lr={backbone_lr:.0e} — already done "
            f"(val_f1={best_val_f1:.4f}, test_f1={test_f1})"
        )
        return best_val_f1, test_f1

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, n_blocks = build_model(num_classes, unfreeze_k)
    model.to(device)
    groups, n_bb, n_hd = param_groups(model, backbone_lr)
    print(
        f"  blocks={n_blocks} unfrozen_last={unfreeze_k}  "
        f"trainable: backbone={n_bb / 1e6:.2f}M  head={n_hd / 1e6:.2f}M"
    )

    total_steps = EPOCHS * len(train_loader)
    opt = torch.optim.AdamW(groups)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    run_name = f"k{unfreeze_k}_bb{backbone_lr:.0e}"
    group_name = f"sweep_k{unfreeze_k}"
    wb_run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=group_name,
        config={
            "unfreeze_k": unfreeze_k,
            "backbone_lr": backbone_lr,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_steps": WARMUP_STEPS,
            "backbone": BACKBONE,
            "n_blocks": n_blocks,
            "trainable_backbone_params": n_bb,
            "trainable_head_params": n_hd,
        },
        tags=["cycle4", "partial_unfreeze", f"k={unfreeze_k}"],
        reinit=True,
        dir="logs/",
    )

    best_val_f1 = -1.0
    best_state = None
    epoch_logs = []
    global_step = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(imgs)
                loss = ce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            global_step += 1
            if (step + 1) % 200 == 0:
                wb_run.log(
                    {
                        "train/loss_step": running_loss / n_seen,
                        "train/lr_backbone": opt.param_groups[0]["lr"],
                        "train/lr_head": opt.param_groups[1]["lr"],
                        "global_step": global_step,
                    }
                )
        train_loss = running_loss / n_seen

        _, _, val_f1 = evaluate(model, val_loader, device)
        dt = time.time() - t0
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "elapsed_s": dt,
            }
        )
        wb_run.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/f1_macro": val_f1,
                "epoch_time_s": dt,
            }
        )
        print(
            f"    ep{epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_f1={val_f1:.4f}  ({dt:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Restore best for test eval
    model.load_state_dict(best_state)
    _, _, test_f1 = evaluate(model, test_loader, device)
    print(f"  best_val_f1={best_val_f1:.4f}  test_f1={test_f1:.4f}")

    wb_run.log(
        {
            "val/best_f1_macro": best_val_f1,
            "test/f1_macro": test_f1,
        }
    )
    wb_run.summary["best_val_f1"] = best_val_f1
    wb_run.summary["test_f1_macro"] = test_f1

    torch.save(
        {
            "state_dict": best_state,
            "config": {
                "unfreeze_k": unfreeze_k,
                "backbone_lr": backbone_lr,
                "head_lr": HEAD_LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "backbone": BACKBONE,
            },
            "best_val_f1": float(best_val_f1),
            "test_f1_macro": float(test_f1),
            "logs": epoch_logs,
        },
        out / "best.ckpt",
    )
    print(f"  [save] {out / 'best.ckpt'}")

    wb_run.finish(quiet=True)

    del model, best_state, opt, sched, scaler
    torch.cuda.empty_cache()

    return float(best_val_f1), float(test_f1)


def main():
    parser = ArgumentParser(
        description="Cycle 4: backbone LR sweep at fixed unfreeze depth"
    )
    parser.add_argument("--unfreeze_k", type=int, required=True, choices=[2, 4])
    parser.add_argument("--wandb_project", default="iwildcam2022_cycle4_sweep")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/cycle4_sweep")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dm = iWildCamDataModule(
        root_dir=ROOT,
        metadata_path=META,
        image_dir="train/train",
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
    dm.setup("fit")
    dm.setup("test")
    num_classes = dm.num_classes
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    print(
        f"num_classes={num_classes}  train_batches={len(train_loader)}  "
        f"val={len(val_loader)}  test={len(test_loader)}"
    )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for backbone_lr in BACKBONE_LR_GRID:
        print(f"\n{'=' * 70}")
        print(f"k={args.unfreeze_k}  backbone_lr={backbone_lr:.0e}")
        print(f"{'=' * 70}")
        val_f1, test_f1 = run_single(
            unfreeze_k=args.unfreeze_k,
            backbone_lr=backbone_lr,
            dm=dm,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        results[backbone_lr] = (val_f1, test_f1)

    print(f"\n{'=' * 70}")
    print(f"CYCLE 4 SWEEP RESULTS — k={args.unfreeze_k}")
    print(f"{'=' * 70}")
    sorted_lrs = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    best_lr, (best_val, best_test) = sorted_lrs[0]
    for lr, (val_f1, test_f1) in sorted(results.items()):
        marker = " <-- best (val_f1)" if lr == best_lr else ""
        test_str = f"{test_f1:.4f}" if test_f1 is not None else "n/a"
        print(f"  bb_lr={lr:.0e}  val_f1={val_f1:.4f}  test_f1={test_str}{marker}")
    print(
        f"\nBest backbone_lr={best_lr:.0e}  "
        f"val_f1={best_val:.4f}  test_f1={best_test:.4f}"
    )


if __name__ == "__main__":
    main()
