"""Condition 2 — K=4 + CE end-to-end training on WILDS.

Reuses the exact training protocol from `src/cycle4_partial_unfreeze.py`:
AdamW, bb_lr=2e-5, head_lr=1e-3, wd=0.05/0.0, 500-step linear warmup,
cosine decay, AMP, batch 32, 5 epochs. K=4 = last 4 transformer blocks +
final LayerNorm trainable. Model selection on WILDS OOD val (`val`) by
eligible-class macro-F1. Saves best checkpoint and per-split predictions
on all 4 eval splits.

Prediction loop tears down each DataLoader (`del` + `gc.collect()` +
`torch.cuda.empty_cache()`) before building the next one — without this,
four WILDS dataset instances end up resident simultaneously by the time
the largest split (OOD test, N=42,791) is reached, and the kernel can kill
the process under memory pressure.
"""

from __future__ import annotations

import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
from _common import (
    BACKBONE,
    CHECKPOINT_DIR,
    EVAL_SPLITS,
    NUM_CLASSES,
    SEED,
    build_partial_unfreeze_model,
    load_segments,
)
from sklearn.metrics import f1_score

from data.wilds import eval_transform, get_loader, train_transform

# Fixed hyperparameters (verbatim from cycle4_partial_unfreeze.py)
UNFREEZE_K = 4
LR_BACKBONE = 2e-5
LR_CLASSIFIER = 1e-3
WEIGHT_DECAY_BB = 0.05
WEIGHT_DECAY_HD = 0.0
WARMUP_STEPS = 500
EPOCHS = 5
BATCH_SIZE = 32
EVAL_BATCH = 64
USE_AMP = True
NUM_WORKERS = 4
EVAL_NUM_WORKERS = 2


def param_groups(model):
    bb_trainable = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return (
        [
            {
                "params": bb_trainable,
                "lr": LR_BACKBONE,
                "weight_decay": WEIGHT_DECAY_BB,
            },
            {
                "params": head_params,
                "lr": LR_CLASSIFIER,
                "weight_decay": WEIGHT_DECAY_HD,
            },
        ],
        sum(p.numel() for p in bb_trainable),
        sum(p.numel() for p in head_params),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def macro_f1_eligible(preds, targets, eligible: set) -> float:
    label_list = sorted(eligible)
    return float(
        f1_score(targets, preds, labels=label_list, average="macro", zero_division=0)
    )


def main() -> None:
    out_dir = CHECKPOINT_DIR / "k4_ce"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  AMP={USE_AMP}")

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    segs = load_segments()
    elig_val = set(segs["eligible_per_split"]["val"])
    elig_per_split = {s: set(segs["eligible_per_split"][s]) for s in EVAL_SPLITS}

    print("Building WILDS loaders…")
    train_loader = get_loader(
        "train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=train_transform(),
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = get_loader(
        "val",
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        transform=eval_transform(),
        num_workers=NUM_WORKERS,
    )
    print(f"  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    model, n_blocks = build_partial_unfreeze_model(NUM_CLASSES, unfreeze_k=UNFREEZE_K)
    model.to(device)
    groups, n_bb, n_hd = param_groups(model)
    print(
        f"  blocks={n_blocks}  unfrozen_last={UNFREEZE_K}  "
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

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    epoch_logs = []

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
            if (step + 1) % 200 == 0:
                print(
                    f"    ep{epoch + 1}  step {step + 1}/{len(train_loader)}  "
                    f"loss={running_loss / n_seen:.4f}  "
                    f"lr_bb={opt.param_groups[0]['lr']:.2e}  "
                    f"lr_hd={opt.param_groups[1]['lr']:.2e}",
                    flush=True,
                )
        train_loss = running_loss / n_seen

        val_preds, val_targets = predict(model, val_loader, device)
        val_f1 = macro_f1_eligible(val_preds, val_targets, elig_val)
        dt = time.time() - t0
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_f1_eligible": val_f1,
                "elapsed_s": dt,
            }
        )
        print(
            f"  ep{epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_eligible_f1={val_f1:.4f}  ({dt:.1f}s)",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"\n[best] epoch {best_epoch}  val_eligible_f1={best_val_f1:.4f}")

    del train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(
        {
            "state_dict": best_state,
            "best_val_f1_eligible": best_val_f1,
            "best_epoch": best_epoch,
            "logs": epoch_logs,
            "config": {
                "unfreeze_k": UNFREEZE_K,
                "lr_backbone": LR_BACKBONE,
                "lr_classifier": LR_CLASSIFIER,
                "weight_decay_bb": WEIGHT_DECAY_BB,
                "weight_decay_hd": WEIGHT_DECAY_HD,
                "warmup_steps": WARMUP_STEPS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "backbone": BACKBONE,
            },
        },
        out_dir / "best.ckpt",
    )

    model.load_state_dict(best_state)
    model.eval()

    print("\nPredicting on all 4 eval splits…")
    preds_by_split: dict[str, np.ndarray] = {}
    for split in EVAL_SPLITS:
        loader = get_loader(
            split,
            batch_size=EVAL_BATCH,
            shuffle=False,
            transform=eval_transform(),
            num_workers=EVAL_NUM_WORKERS,
            pin_memory=False,
        )
        preds, targets = predict(model, loader, device)
        f1 = macro_f1_eligible(preds, targets, elig_per_split[split])
        preds_by_split[split] = preds
        print(f"  {split:8s} eligible-macro-F1={f1:.4f}  N={len(targets)}", flush=True)

        del loader
        gc.collect()
        torch.cuda.empty_cache()

    np.savez(
        out_dir / "predictions.npz",
        **{f"{split}_preds": preds_by_split[split] for split in EVAL_SPLITS},
    )
    print(f"\n[save] {out_dir / 'best.ckpt'}")
    print(f"[save] {out_dir / 'predictions.npz'}")


if __name__ == "__main__":
    main()
