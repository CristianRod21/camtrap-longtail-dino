# Cycle 1 — Full-Frame Baseline

**Research question**: What is the classification performance of standard backbones on iWildCam when using full, uncropped frames as input?

Cycle 1 is the baseline. Images are resized directly to 224×224 and fed to the model.

---

## Dataset setup

### 1. Download

```bash
kaggle competitions download -c iwildcam2022-fgvc9
```

### 2. Extract

```bash
python src/scripts/extract_dataset.py \
    /path/to/iwildcam2022-fgvc9.zip \
    /path/to/iwildcam/
```

### 3. Preprocess

Resizes all images to 224×224 squares so every model sees the same input resolution regardless of the original aspect ratio.

```bash
python src/scripts/preprocess_dataset.py \
    --input_dir /path/to/iwildcam \
    --output_dir /path/to/iwildcam_224 \
    --size 224 \
    --num_workers 16
```

Update `base_dir` in `src/configs/dataset/iwildcam.yaml` to point to the preprocessed directory before training.

---

## Models

| Model | Config | Protocol |
|---|---|---|
| DINOv2 ViT-B/14 | `cycle_1_thesis/dinov2_vitb14.yaml` | Linear eval (frozen) |
| ResNet50 pretrained | `cycle_1_thesis/resnet50_pretrained.yaml` | Linear eval (frozen) |
| timm ViT-B/16 | `cycle_1_thesis/timm_vit_base_patch16_224.yaml` | Linear eval (frozen) |
| ResNet50 scratch | `cycle_1_thesis/resnet50_scratch.yaml` | Full train from scratch |

All configs use `batch_size: 64`, `num_workers: 4`.

---

## Step 1 — Linear eval sweep

Sweeps 13 LRs from `1e-5` to `0.1`. Features are cached once and reused across all LRs.

Run from the project root. Run sequentially — each job saturates the GPU.

```bash
# DINOv2 ViT-B/14
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_1_thesis/dinov2_vitb14.yaml \
    --wandb_project iwildcam2022_cycle1_sweep_test \
    --checkpoint_dir checkpoints/cycle1_sweep

# ResNet50 pretrained
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_1_thesis/resnet50_pretrained.yaml \
    --wandb_project iwildcam2022_cycle1_sweep_test \
    --checkpoint_dir checkpoints/cycle1_sweep

# timm ViT-B/16
uv run src/linear_eval_sweep.py \
    --config src/configs/model/cycle_1_thesis/timm_vit_base_patch16_224.yaml \
    --wandb_project iwildcam2022_cycle1_sweep_test \
    --checkpoint_dir checkpoints/cycle1_sweep
```

Results land in `checkpoints/cycle1_sweep/sweep_<model>/results.yaml`.

---

## Step 2 — ResNet50 from scratch

Probe phase: 5 LRs × 10 epochs → picks winner → 50-epoch full run.

```bash
uv run src/train_scratch.py \
    --config src/configs/model/cycle_1_thesis/resnet50_scratch.yaml \
    --wandb_project iwildcam_cycle1_sweep \
    --checkpoint_dir checkpoints/cycle1_sweep
```

---

## Hardware

Runs were done on 24 GB VRAM. If you're on a smaller GPU, reduce `batch_size` in the model config — 32 or 16 should work, though it may affect results slightly.