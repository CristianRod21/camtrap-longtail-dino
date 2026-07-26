import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Serengeti(Dataset):
    def __init__(self, data_dir, split_name, transform=None):
        """
        Args:
            data_dir (str): Directory containing the memmap dataset files.
            split_name (str): Split prefix, e.g. 'train', 'val', 'test'.
            transform (callable, optional): Transform to apply to the image.
        """
        self.data_dir = data_dir
        self.split_name = split_name
        self.transform = transform

        # Load metadata
        with open(os.path.join(data_dir, f"{split_name}_meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)

        # Memory-mapped arrays
        self.images = np.memmap(
            os.path.join(data_dir, f"{split_name}_images.bin"),
            dtype=np.uint8,
            mode="r",
        )
        self.labels = np.memmap(
            os.path.join(data_dir, f"{split_name}_labels.bin"),
            dtype=np.int64,
            mode="r",
        )

    def __len__(self):
        return self.meta["num_images"]

    def _load_image(self, idx):
        C, H, W = self.meta["image_shape"]
        offset = self.meta["image_offsets"][idx]

        img = (
            self.images[offset : offset + C * H * W]
            .copy()
            .reshape(C, H, W)
            .transpose(1, 2, 0)
        )

        return Image.fromarray(img, mode="RGB")

    def _load_label(self, idx):
        return int(self.labels[self.meta["label_offsets"][idx]])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self._load_image(idx)
        label = self._load_label(idx)

        if self.transform:
            image = self.transform(image)

        return image, label


class SerengetiDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

        self.transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.transform_val = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self):
        # Assume files already exist
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = Serengeti(
                data_dir=self.root_dir,
                split_name="train",
                transform=self.transform_train,
            )
            full_val = Serengeti(
                data_dir=self.root_dir,
                split_name="val",
                transform=self.transform_val,
            )

            train_labels = [full_train._load_label(i) for i in range(len(full_train))]
            val_labels = [full_val._load_label(i) for i in range(len(full_val))]

            self.train_class_dist = {}
            for y in train_labels:
                self.train_class_dist[y] = self.train_class_dist.get(y, 0) + 1

            self.val_class_dist = {}
            for y in val_labels:
                self.val_class_dist[y] = self.val_class_dist.get(y, 0) + 1

            self.train_dataset = full_train
            self.val_dataset = full_val

            present_classes = np.unique(train_labels)

            weights_present = compute_class_weight(
                class_weight="balanced",
                classes=present_classes,
                y=train_labels,
            )

            num_classes = int(max(present_classes)) + 1
            self.num_classes = num_classes
            full_class_weights = np.zeros(num_classes, dtype=np.float32)
            full_class_weights[present_classes] = weights_present
            self.class_weights = torch.tensor(full_class_weights, dtype=torch.float32)

        if stage == "test" or stage is None:
            self.test_dataset = Serengeti(
                data_dir=self.root_dir,
                split_name="test",
                transform=self.transform_val,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
