# Cycle 2 — MegaDetector Cropping

**Research question**: Does cropping images to the MegaDetector-detected animal region, instead of feeding the full frame, improve classification performance across backbones?

Same models, splits, and LR grid as Cycle 1 — only the input crop changes, to isolate its effect.

---

## What changed vs. Cycle 1

| | Cycle 1 | Cycle 2 |
|---|---|---|
| Input | Full frame, resized to 224×224 | MegaDetector crop, resized to 224×224 |
| Dataset config | `iwildcam.yaml` | `iwildcam_cropped.yaml` |
| Model configs | `cycle_1_thesis/` | `cycle_2_thesis/` |
| Checkpoint dir | `checkpoints/cycle1_sweep` | `checkpoints/cycle2_sweep` |

Use a distinct `--checkpoint_dir` for every cycle — Cycle 1's checkpoints were
overwritten once already by a Cycle 2 run reusing the same directory in an
earlier version of this pipeline.

---

## Step 0 — MegaDetector crop lookup

Cropping is applied inside `iWildCam.__getitem__` (`src/data/iwildcam.py`) from a
precomputed bbox lookup: filename → highest-confidence animal detection, normalized
`[x1, y1, x2, y2]`. Images absent from the lookup fall back to the full frame.

The lookup already exists at `<iwildcam_224 base_dir>/megadetector_train.json`
(106,244 / 201,399 train images have a detection above threshold). To regenerate it:

```bash
uv run src/data/run_megadetector.py \
    --image_dir /path/to/iwildcam_224/train/train \
    --output_json /path/to/iwildcam_224/megadetector_train.json \
    --batch_size 64 --device cuda
```

`src/configs/dataset/iwildcam_cropped.yaml` points both the `train` and `val` split
entries at `megadetector_train.json` — this pipeline builds train/val/test all as an
internal split of the same `train/train` metadata (see `iWildCamDataModule._split_dataset`),
so there is only one physical image pool and one detections file.

---

## Models

| Model | Config | Protocol |
|---|---|---|
| DINOv2 ViT-B/14 | `cycle_2_thesis/dinov2_vitb14.yaml` | Linear eval (frozen) |
| ResNet50 pretrained | `cycle_2_thesis/resnet50_pretrained.yaml` | Linear eval (frozen) |
| timm ViT-B/16 | `cycle_2_thesis/timm_vit_base_patch16_224.yaml` | Linear eval (frozen) |
| ResNet50 scratch | `cycle_2_thesis/resnet50_scratch.yaml` | Full train from scratch |

All configs use `batch_size: 64`, `num_workers: 4`.

---

## Step 1 — Linear eval sweep

Sweeps 13 LRs from `1e-5` to `0.1`. Features are cached once and reused across all LRs.

Run from the project root. Run sequentially — each job saturates the GPU.

```bash
# DINOv2 ViT-B/14
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_2_thesis/dinov2_vitb14.yaml \
    --wandb_project iwildcam2022_cycle2_sweep \
    --checkpoint_dir checkpoints/cycle2_sweep

# ResNet50 pretrained
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_2_thesis/resnet50_pretrained.yaml \
    --wandb_project iwildcam2022_cycle2_sweep \
    --checkpoint_dir checkpoints/cycle2_sweep

# timm ViT-B/16
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_2_thesis/timm_vit_base_patch16_224.yaml \
    --wandb_project iwildcam2022_cycle2_sweep \
    --checkpoint_dir checkpoints/cycle2_sweep
```

Results land in `checkpoints/cycle2_sweep/sweep_<model>/results.yaml`.

---

## Step 2 — ResNet50 from scratch

Probe phase: 5 LRs × 10 epochs → picks winner → 50-epoch full run.

```bash
uv run src/train_scratch.py \
    --config src/configs/model/cycle_2_thesis/resnet50_scratch.yaml \
    --wandb_project iwildcam2022_cycle2_sweep \
    --checkpoint_dir checkpoints/cycle2_sweep
```

---

## Hardware

Runs were done on 24 GB VRAM. If you're on a smaller GPU, reduce `batch_size` in the model config — 32 or 16 should work, though it may affect results slightly.
