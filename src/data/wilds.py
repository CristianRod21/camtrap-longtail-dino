"""WILDS iWildCam (v2.0) data adapters.

The WILDS subset's __getitem__ returns `(image, label, metadata)`. For
end-to-end training we need plain `(image, label)` pairs (CE loss only);
for eval-time bootstrap we need the metadata (location id at column 0).
Two thin wrappers handle both.

Data is auto-downloaded on first use via the `wilds` package (~11GB
compressed, ~12GB extracted) if not already present at WILDS_ROOT.
"""

from __future__ import annotations

from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

WILDS_ROOT = Path("/path/to/wilds")

# Standard ImageNet normalization (matches the iWildCam 2022 pipeline)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def eval_transform():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def get_dataset():
    from wilds import get_dataset as _get

    return _get(dataset="iwildcam", download=True, root_dir=str(WILDS_ROOT))


class _DropMeta(Dataset):
    """Wrap a WILDS subset so __getitem__ returns just (x, y)."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y, _meta = self.subset[idx]
        return x, y


def get_loader(
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    transform,
    num_workers: int = 4,
    drop_last: bool = False,
    drop_meta: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a DataLoader for a WILDS split.

    `drop_meta=True` makes the loader yield `(x, y)` (used for training and
    fast eval). `drop_meta=False` keeps the WILDS `(x, y, metadata)` tuple.
    """
    dataset = get_dataset()
    subset = dataset.get_subset(split, transform=transform)
    if drop_meta:
        subset = _DropMeta(subset)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
