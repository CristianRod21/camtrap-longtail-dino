import json
import os
from collections import Counter, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class iWildCam(Dataset):
    def __init__(self, metadata, image_dir, transform=None, bbox_lookup=None):
        """
        Args:
            metadata (string): Path to the JSON file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a
                sample.
            bbox_lookup (dict, optional): MegaDetector detections dict mapping
                filename -> [x1_norm, y1_norm, x2_norm, y2_norm].  When provided,
                images are cropped to the highest-confidence animal detection before
                the transform is applied.  Images absent from the dict fall back to
                the full image.
        """
        with open(metadata) as f:
            self.metadata = json.load(f)

        self.categories = self.metadata["categories"]
        self.images = self.metadata["images"]
        self.annotations = self.metadata["annotations"]
        self.image_dir = image_dir
        self.transform = transform
        self.bbox_lookup = bbox_lookup or {}

        self.image_lookup = {img["id"]: img for img in self.images}

        self.image_to_annotation = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_to_annotation:
                self.image_to_annotation[image_id] = []
            self.image_to_annotation[image_id].append(ann)

        self.category_to_idx = {
            item["id"]: idx for idx, item in enumerate(self.categories)
        }

        self.valid_image_ids = list(self.image_to_annotation.keys())

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.valid_image_ids[idx]
        image_meta = self.image_lookup[image_id]
        annotations = self.image_to_annotation[image_id]

        # Use the first annotation's category as the label
        category_id = annotations[0]["category_id"]
        label = self.category_to_idx[category_id]

        img_name = os.path.join(self.image_dir, image_meta["file_name"])

        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        # MegaDetector crop — uses pre-computed bbox if available
        filename = os.path.basename(img_name)
        bbox = self.bbox_lookup.get(filename)
        if bbox is not None:
            w, h = image.size
            x1_n, y1_n, x2_n, y2_n = bbox
            x1, y1, x2, y2 = int(x1_n * w), int(y1_n * h), int(x2_n * w), int(y2_n * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        return image, label


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a Subset and applies a transform without mutating the parent dataset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    @property
    def dataset(self):
        return self.subset.dataset

    @property
    def indices(self):
        return self.subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class iWildCamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        metadata_path,
        image_dir,
        batch_size=32,
        num_workers=4,
        train_transform=None,
        val_transform=None,
        train_detections_path=None,
        val_detections_path=None,
    ):
        super().__init__()
        self.metadata_path = root_dir / metadata_path
        self.image_dir = root_dir / image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Load MegaDetector bbox lookups (Cycle 2 — optional)
        self.train_bbox_lookup = self._load_detections(train_detections_path)
        self.val_bbox_lookup = self._load_detections(val_detections_path)

        # Define transforms (caller-supplied transforms take priority)
        self.transform_train = train_transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_val = val_transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        with open(self.metadata_path) as f:
            metadata = json.load(f)
        self.num_classes = len(metadata["categories"])

        self.train_indices = None
        self.val_indices = None

    @staticmethod
    def _load_detections(path) -> dict:
        if path is None:
            return {}
        with open(path) as f:
            return json.load(f)

    def prepare_data(self):
        # Nothing to download - assuming data is already available
        pass

    def _class_dist(self, dataset, indices):
        dist = {}
        for idx in indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            dist[label] = dist.get(label, 0) + 1
        return dist

    def _stratified_val_split(
        self, dataset, train_indices, val_frac=0.10, min_class_count=10, random_state=42
    ):
        """Take a 10% stratified subsample from training for validation.

        Classes with fewer than *min_class_count* samples are kept entirely
        in training — they're too rare to spare any for monitoring.
        """
        rng = np.random.RandomState(random_state)

        # Group train indices by class label
        class_to_indices = defaultdict(list)
        for idx in train_indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            class_to_indices[label].append(idx)

        final_train = []
        final_val = []

        for label, indices in class_to_indices.items():
            if len(indices) < min_class_count:
                final_train.extend(indices)
                continue
            n_val = max(1, int(len(indices) * val_frac))
            shuffled = list(indices)
            rng.shuffle(shuffled)
            final_val.extend(shuffled[:n_val])
            final_train.extend(shuffled[n_val:])

        print(
            f"Stratified val split: {len(final_train)} train, {len(final_val)} val "
            f"(classes <{min_class_count} samples kept train-only)"
        )
        return final_train, final_val

    def setup(self, stage=None):
        full_dataset = iWildCam(
            metadata=self.metadata_path,
            image_dir=self.image_dir,
            transform=None,
            bbox_lookup={**self.train_bbox_lookup, **self.val_bbox_lookup},
        )

        # Location-separated split → train pool + test set
        train_pool_indices, test_indices = self._split_dataset(full_dataset)

        # Stratified 10% of train pool → small val set for monitoring
        train_indices, val_indices = self._stratified_val_split(
            full_dataset, train_pool_indices
        )

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Collect class distributions for all three splits
        self.train_class_dist = self._class_dist(full_dataset, train_indices)
        self.val_class_dist = self._class_dist(full_dataset, val_indices)
        self.test_class_dist = self._class_dist(full_dataset, test_indices)

        if stage == "fit" or stage is None:
            self.train_dataset = self._create_subset(
                full_dataset, train_indices, self.transform_train
            )
            self.val_dataset = self._create_subset(
                full_dataset, val_indices, self.transform_val
            )

            # Compute class weights from training set
            train_labels = []
            for idx in train_indices:
                img_id = full_dataset.valid_image_ids[idx]
                ann = full_dataset.image_to_annotation[img_id][0]
                label = full_dataset.category_to_idx[ann["category_id"]]
                train_labels.append(label)

            present_classes = np.unique(train_labels)
            weights_present = compute_class_weight(
                class_weight="balanced", classes=present_classes, y=train_labels
            )

            full_class_weights = np.zeros(self.num_classes, dtype=np.float32)
            full_class_weights[present_classes] = weights_present
            self.class_weights = torch.tensor(full_class_weights)

        if stage == "test" or stage is None:
            self.test_dataset = self._create_subset(
                full_dataset, test_indices, self.transform_val
            )

    def _split_dataset(
        self,
        dataset,
        val_size=0.2,
        min_class_coverage=0.80,
        max_val_size=0.3,
        random_state=42,
    ):
        """
        Creates a hybrid split with mostly different locations but shared classes:
        • Separates locations between train and val where possible
        • Ensures good class coverage in both train and val
        • Maintains reasonable train/val ratio
        """
        import numpy as np

        rng = np.random.RandomState(random_state)

        # 1. Gather metadata
        loc_to_indices = defaultdict(list)
        loc_to_classes = defaultdict(set)
        class_to_locs = defaultdict(set)
        class_counts = Counter()

        for idx, img_id in enumerate(dataset.valid_image_ids):
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            loc = dataset.image_lookup[img_id]["location"]

            loc_to_indices[loc].append(idx)
            loc_to_classes[loc].add(label)
            class_to_locs[label].add(loc)
            class_counts[label] += 1

        all_classes = set(class_to_locs.keys())
        total_samples = len(dataset.valid_image_ids)

        # 2. Find rare classes
        rare_classes = {cls for cls, count in class_counts.items() if count <= 5}
        print(f"Found {len(rare_classes)} rare classes with ≤5 samples")

        # 3. Initial approach: try to ensure all classes in both sets
        # Create a pool of locations we must share between train and val
        shared_locs = set()

        # For each class, find the locations that have it
        for cls in all_classes:
            locs_with_class = list(class_to_locs[cls])

            if len(locs_with_class) == 1:
                # If class appears in only one location, we must share it
                shared_locs.add(locs_with_class[0])
            elif len(locs_with_class) <= 3:
                # If class appears in only a few locations, we should share at least one
                rng.shuffle(locs_with_class)
                shared_locs.add(locs_with_class[0])

        print(
            f"Identified {len(shared_locs)} locations that should be shared "
            "between train and test"
        )

        non_shared_locs = set(loc_to_indices.keys()) - shared_locs
        print(
            f"Remaining {len(non_shared_locs)} locations can be split "
            "between train and test"
        )

        # 4. Create location assignments
        non_shared_list = list(non_shared_locs)
        rng.shuffle(non_shared_list)

        shared_samples = sum(len(loc_to_indices[loc]) for loc in shared_locs)

        target_test_ratio = min(val_size * total_samples, max_val_size * total_samples)
        target_test_non_shared = max(0, target_test_ratio - shared_samples * val_size)

        test_only_locs = set()
        train_only_locs = set()

        current_test_samples = 0
        for loc in non_shared_list:
            if current_test_samples >= target_test_non_shared:
                train_only_locs.add(loc)
            else:
                test_only_locs.add(loc)
                current_test_samples += len(loc_to_indices[loc])

        # 5. For shared locations, split the samples
        shared_train_indices = []
        shared_test_indices = []

        for loc in shared_locs:
            shuffled_samples = list(loc_to_indices[loc])
            rng.shuffle(shuffled_samples)
            split_idx = int(len(shuffled_samples) * 0.7)
            shared_train_indices.extend(shuffled_samples[:split_idx])
            shared_test_indices.extend(shuffled_samples[split_idx:])

        # 6. Build final index lists
        train_indices = shared_train_indices + [
            idx for loc in train_only_locs for idx in loc_to_indices[loc]
        ]
        test_indices = shared_test_indices + [
            idx for loc in test_only_locs for idx in loc_to_indices[loc]
        ]

        # 7. Calculate final statistics
        train_classes = set()
        for idx in train_indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            train_classes.add(label)

        test_classes = set()
        for idx in test_indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            test_classes.add(label)

        common_classes = train_classes.intersection(test_classes)
        missing_from_train = all_classes - train_classes
        missing_from_test = all_classes - test_classes

        print("Final hybrid split with shared locations:")
        print(
            f"  • {len(train_indices)} train imgs, {len(test_indices)} test imgs "
            f"({len(train_indices) / total_samples:.1%}/"
            f"{len(test_indices) / total_samples:.1%})"
        )
        print("  • Locations:")
        print(f"    - {len(train_only_locs)} train-only")
        print(f"    - {len(test_only_locs)} test-only")
        print(f"    - {len(shared_locs)} shared")
        print(
            f"  • Classes in both splits: {len(common_classes)}/{len(all_classes)} "
            f"({len(common_classes) / len(all_classes):.1%})"
        )
        print(f"  • Classes missing from train: {len(missing_from_train)}")
        print(f"  • Classes missing from test: {len(missing_from_test)}")

        # Calculate JS distances for analysis
        train_class_dist = Counter()
        test_class_dist = Counter()

        for idx in train_indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            train_class_dist[label] += 1

        for idx in test_indices:
            img_id = dataset.valid_image_ids[idx]
            ann = dataset.image_to_annotation[img_id][0]
            label = dataset.category_to_idx[ann["category_id"]]
            test_class_dist[label] += 1

        from scipy.spatial.distance import jensenshannon

        all_classes_list = sorted(all_classes)
        train_probs = np.array([train_class_dist.get(c, 0) for c in all_classes_list])
        test_probs = np.array([test_class_dist.get(c, 0) for c in all_classes_list])

        if sum(train_probs) > 0:
            train_probs = train_probs / sum(train_probs)
        if sum(test_probs) > 0:
            test_probs = test_probs / sum(test_probs)

        js_dist = jensenshannon(train_probs, test_probs)
        print(f"Class distribution JS distance: {js_dist:.4f}")

        loc_overlap_pct = len(shared_locs) / (
            len(train_only_locs) + len(test_only_locs) + len(shared_locs)
        )
        print(f"Location overlap: {loc_overlap_pct:.1%}")

        return train_indices, test_indices

    def _create_subset(self, dataset, indices, transform):
        subset = torch.utils.data.Subset(dataset, indices)
        return _SubsetWithTransform(subset, transform)

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
