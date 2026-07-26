"""Extract a Kaggle competition zip archive in parallel.

Usage:
    python extract_dataset.py <zip_path> <extract_dir> [--workers N]

Download the archive first with:
    kaggle competitions download -c iwildcam2022-fgvc9
"""

import argparse
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor


def extract_parallel(zip_path, extract_dir, num_workers=4):
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files = zip_ref.namelist()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(zip_ref.extract, file, extract_dir) for file in files
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zip_path", help="Path to the downloaded .zip file")
    parser.add_argument("extract_dir", help="Directory to extract files into")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )
    args = parser.parse_args()

    extract_parallel(args.zip_path, args.extract_dir, num_workers=args.workers)
