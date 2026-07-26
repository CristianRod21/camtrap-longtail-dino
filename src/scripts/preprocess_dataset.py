import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def process_single_image(args):
    input_path, output_path, size = args
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(output_path)
    except Exception as e:
        print(f"Failed: {input_path}: {e}")


def preprocess_iwildcam(
    input_dir: str,
    output_dir: str,
    size: int,
    num_workers: int,
    splits: list[str],
    metadata: str = "metadata",
):
    """Copy metadata and resize images in specified splits."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy metadata only
    meta_src = input_path / metadata
    meta_dst = output_path / metadata
    if meta_src.exists():
        print(f"Copying {metadata}/...")
        shutil.copytree(meta_src, meta_dst)

    # Resize images in each split
    for split in splits:
        split_dir = input_path / split / split
        out_split = output_path / split / split
        out_split.mkdir(parents=True, exist_ok=True)

        image_args = []
        for file in os.listdir(split_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_args.append((str(split_dir / file), str(out_split / file), size))

        if not image_args:
            print(f"No images found in {split_dir}")
            continue

        print(f"Resizing {len(image_args)} images in {split}/...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(process_single_image, image_args),
                    total=len(image_args),
                    desc=split,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="dataset/iwildcam_v2.0")
    parser.add_argument("--output_dir", default="dataset_preprocessed/iwildcam_v2.0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--metadata", default="metadata")
    args = parser.parse_args()

    preprocess_iwildcam(
        args.input_dir,
        args.output_dir,
        args.size,
        args.num_workers,
        args.splits,
        args.metadata,
    )
