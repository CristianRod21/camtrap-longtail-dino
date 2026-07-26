"""
Offline MegaDetector preprocessing for iWildCam (Cycle 2).

Runs MegaDetectorV5a on all images in a folder and saves a JSON lookup:
    { "filename.jpg": [x1_norm, y1_norm, x2_norm, y2_norm] }

Only the highest-confidence *animal* detection (class 0) per image is kept.
Images with no animal detection above threshold are omitted — the dataset
falls back to the full image for those.

Usage:
    # Train split
    uv run src/data/run_megadetector.py \\
        --image_dir /path/to/iwildcam_224/train/train \\
        --output_json /path/to/iwildcam_224/megadetector_train.json \\
        --batch_size 64 --device cuda

    # Val split
    uv run src/data/run_megadetector.py \\
        --image_dir /path/to/iwildcam_224/val/val \\
        --output_json /path/to/iwildcam_224/megadetector_val.json \\
        --batch_size 64 --device cuda
"""

import argparse
import json
from pathlib import Path

from PytorchWildlife.models import detection as pw_det

ANIMAL_CLASS_ID = 0
DEFAULT_CONF_THRESHOLD = 0.2  # MegaDetectorV5 default


def run(
    image_dir: str,
    output_json: str,
    batch_size: int,
    device: str,
    conf_threshold: float,
):
    image_dir = Path(image_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MegaDetectorV5a on {device}...")
    md = pw_det.MegaDetectorV5(device=device, pretrained=True, version="a")

    print(
        f"Running detection on {image_dir} "
        f"(batch_size={batch_size}, conf_threshold={conf_threshold})..."
    )
    results = md.batch_image_detection(
        str(image_dir),
        batch_size=batch_size,
        det_conf_thres=conf_threshold,
    )

    # Build lookup: filename -> [x1_norm, y1_norm, x2_norm, y2_norm]
    # 'normalized_coords' in results is [[x1/w, y1/h, x2/w, y2/h], ...]
    lookup = {}
    for entry in results:
        img_path = entry["img_id"]
        filename = Path(img_path).name

        detections = entry["detections"]
        norm_coords = entry["normalized_coords"]

        if len(detections) == 0:
            continue

        # Filter to animal class only
        animal_mask = detections.class_id == ANIMAL_CLASS_ID
        if not animal_mask.any():
            continue

        animal_confs = detections.confidence[animal_mask]
        animal_coords = [c for c, m in zip(norm_coords, animal_mask) if m]

        best_idx = int(animal_confs.argmax())
        lookup[filename] = animal_coords[best_idx]  # [x1_n, y1_n, x2_n, y2_n]

    total_images = sum(1 for _ in image_dir.glob("*.jpg"))
    total_images = max(total_images, len(results))  # fallback count
    print(
        f"Animal detections found: {len(lookup)} / ~{total_images} images "
        f"({len(lookup) / max(total_images, 1):.1%})"
    )

    with open(output_json, "w") as f:
        json.dump(lookup, f)
    print(f"Saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MegaDetectorV5a on an image folder"
    )
    parser.add_argument(
        "--image_dir", required=True, help="Folder of images to process"
    )
    parser.add_argument(
        "--output_json", required=True, help="Path to write detections JSON"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    args = parser.parse_args()

    run(
        args.image_dir,
        args.output_json,
        args.batch_size,
        args.device,
        args.conf_threshold,
    )


if __name__ == "__main__":
    main()
