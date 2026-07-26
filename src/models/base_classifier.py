from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision

from .freeze import freeze_model_layers


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        frozen: bool,
        num_classes: int,
        lr: float,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        batch_size: int = 256,
        freeze_strategy: Literal["all", "last_layer"] = "all",
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize the BaseClassifier.

        Supports two training modes, selected automatically based on
        whether the backbone is fully frozen:

        LINEAR EVAL (frozen=True, freeze_strategy="all"):
          Standard protocol from DINO/DINOv2 (Caron et al., 2021; Oquab et al., 2023):
          - Single linear layer on frozen features (no projection, no dropout)
          - BatchNorm before the classifier for training stability
            (Park et al., 2023 — "Rethinking Evaluation Protocols", arXiv:2304.03456)
          - SGD with momentum 0.9 and no weight decay
            (facebookresearch/dino eval_linear.py)
          - Cosine annealing LR schedule
          - Linear LR scaling rule: effective_lr = lr * batch_size / 256

        FINE-TUNING (frozen=True + freeze_strategy="last_layer", or frozen=False):
          - Same BN + Linear head
          - AdamW with differential LR (backbone: lr*0.01, head: lr)
          - Cosine annealing LR schedule

        Args:
            backbone: Model backbone architecture (resnet18, resnet50, etc.)
            frozen: Whether to freeze backbone layers
            num_classes: Number of output classes
            lr: Base learning rate (scaled by batch_size/256 in linear eval mode)
            weight_decay: Weight decay (default 0.0 for linear eval)
            max_epochs: Maximum number of training epochs
            batch_size: Batch size (used for LR scaling in linear eval mode)
            freeze_strategy: How to freeze the backbone when frozen=True.
                - "all": Freeze the entire backbone (linear eval).
                - "last_layer": Keep the last block trainable (fine-tuning).
            use_gradient_checkpointing: Whether to use gradient checkpointing

        Raises:
            ValueError: If an unsupported backbone is specified
        """
        super().__init__()
        self.save_hyperparameters()

        self.backbone = self._initialize_backbone(backbone)
        self.feature_dim = self._get_feature_dim(self.backbone)

        if use_gradient_checkpointing and hasattr(
            self.backbone, "gradient_checkpointing_enable"
        ):
            self.backbone.gradient_checkpointing_enable()

        self.backbone = freeze_model_layers(
            self.backbone, frozen, strategy=freeze_strategy
        )

        # When the backbone is fully frozen, set it to eval mode so that
        # internal BatchNorm/Dropout layers don't update running stats.
        # Without this, BN layers inside the backbone silently corrupt
        # their running_mean/running_var during training.
        if self.backbone_is_frozen:
            self.backbone.eval()

        # Remove the final classification layer so the backbone outputs raw features.
        # Must happen AFTER _get_feature_dim (which inspects the original head)
        # and AFTER freezing (so freeze code can reference the real head).
        self._remove_backbone_head()

        # Linear evaluation head following the standard protocol:
        # - BatchNorm before the linear layer stabilizes training and
        #   resolves inconsistencies between k-NN and linear probe metrics
        #   (Park et al., 2023 — arXiv:2304.03456)
        # - Single Linear layer, no projection, no dropout
        #   (Oquab et al., 2023 — DINOv2; Radford et al., 2021 — CLIP)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes),
        )

        self.num_classes = num_classes
        self.criterion = F.cross_entropy

        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = torchmetrics.classification.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = torchmetrics.classification.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average=None,  # per-class accuracies
        )

    # Supported torchvision backbones. Class-level constant so subclasses
    # can extend it without overriding _initialize_backbone.
    SUPPORTED_BACKBONES = {
        "resnet18": (
            torchvision.models.resnet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
        ),
        "resnet34": (
            torchvision.models.resnet34,
            torchvision.models.ResNet34_Weights.DEFAULT,
        ),
        "resnet50": (
            torchvision.models.resnet50,
            torchvision.models.ResNet50_Weights.DEFAULT,
        ),
        "resnet101": (
            torchvision.models.resnet101,
            torchvision.models.ResNet101_Weights.DEFAULT,
        ),
        "convnext_tiny": (
            torchvision.models.convnext_tiny,
            torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
        ),
        "convnext_base": (
            torchvision.models.convnext_base,
            torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
        ),
        "efficientnet_v2_s": (
            torchvision.models.efficientnet_v2_s,
            torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
        ),
        "efficientnet_v2_m": (
            torchvision.models.efficientnet_v2_m,
            torchvision.models.EfficientNet_V2_M_Weights.DEFAULT,
        ),
        "swin_t": (
            torchvision.models.swin_t,
            torchvision.models.Swin_T_Weights.DEFAULT,
        ),
        "swin_b": (
            torchvision.models.swin_b,
            torchvision.models.Swin_B_Weights.DEFAULT,
        ),
        "vit_b_16": (
            torchvision.models.vit_b_16,
            torchvision.models.ViT_B_16_Weights.DEFAULT,
        ),
    }

    def _initialize_backbone(self, backbone: str) -> nn.Module:
        if backbone.startswith("dinov2_"):
            try:
                return torch.hub.load("facebookresearch/dinov2", backbone)
            except Exception as e:
                raise RuntimeError(
                    f"Error initializing DINOv2 backbone '{backbone}': {e}"
                ) from e

        if backbone.startswith("timm_"):
            try:
                import timm

                model_name = backbone.removeprefix("timm_")
                return timm.create_model(model_name, pretrained=True)
            except Exception as e:
                raise RuntimeError(
                    f"Error initializing timm backbone '{backbone}': {e}"
                ) from e

        if backbone not in self.SUPPORTED_BACKBONES:
            supported = list(self.SUPPORTED_BACKBONES.keys()) + [
                "dinov2_*",
                "timm_*",
            ]
            raise ValueError(
                f"Backbone '{backbone}' not supported. Options: {supported}"
            )

        model_fn, weights = self.SUPPORTED_BACKBONES[backbone]
        try:
            return model_fn(weights=weights)
        except Exception as e:
            raise RuntimeError(f"Error initializing backbone '{backbone}': {e}") from e

    @staticmethod
    def _get_feature_dim(model: nn.Module) -> int:
        """
        Extract the feature dimension from the backbone's final layer.

        Checks attributes in order of specificity so that e.g. DINOv2's
        embed_dim is picked up before falling through to .fc / .classifier.
        """
        # DINOv2 / DINOv3 — exposes embed_dim directly
        if hasattr(model, "embed_dim"):
            return model.embed_dim

        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            return model.fc.in_features

        if hasattr(model, "classifier"):
            clf = model.classifier
            if isinstance(clf, nn.Sequential):
                for layer in reversed(clf):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(clf, nn.Linear):
                return clf.in_features

        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            return model.head.in_features

        # torchvision ViT — heads.head
        if hasattr(model, "heads"):
            heads = model.heads
            if isinstance(heads, nn.Sequential):
                for layer in reversed(heads):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(heads, nn.Linear):
                return heads.in_features

        raise AttributeError(
            f"Could not determine feature dimension for {model.__class__.__name__}. "
            f"Add explicit handling for this architecture."
        )

    def _remove_backbone_head(self) -> None:
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, "heads"):
            self.backbone.heads = nn.Identity()

    def train(self, mode: bool = True) -> "BaseClassifier":
        """Override to keep the frozen backbone in eval mode.

        Lightning calls model.train() at the start of each training epoch,
        which would flip the backbone's internal BatchNorm/Dropout back to
        train mode. We prevent that here.
        """
        super().train(mode)
        if mode and self.backbone_is_frozen:
            self.backbone.eval()
        return self

    @property
    def backbone_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.backbone.parameters())

    @torch.no_grad()
    def precompute_features(self, dataloader) -> "torch.utils.data.TensorDataset":
        """
        Run the frozen backbone once over a full dataloader and return
        a TensorDataset of (features, labels).

        This lets you skip the backbone entirely during training when
        strategy="all" and frozen=True. Only the classifier head trains.

        Args:
            dataloader: A DataLoader yielding (images, labels) batches.

        Returns:
            TensorDataset of (features [N, feature_dim], labels [N]).

        Raises:
            RuntimeError: If the backbone has trainable parameters.
        """
        if not self.backbone_is_frozen:
            raise RuntimeError(
                "precompute_features requires a fully frozen backbone. "
                "Set frozen=True and freeze_strategy='all'."
            )

        device = next(self.parameters()).device
        self.eval()

        all_features = []
        all_labels = []

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            features = self.extract_features(imgs)
            all_features.append(features.cpu())
            all_labels.append(labels)

        return torch.utils.data.TensorDataset(
            torch.cat(all_features), torch.cat(all_labels)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.

        Returns:
            Tensor of shape [B, D] where D is self.feature_dim.
        """
        # DINOv2 / DINOv3 — use dedicated API for intermediate layers
        if hasattr(self.backbone, "get_intermediate_layers"):
            features = self.backbone.get_intermediate_layers(x, n=1)[0]
            return features[:, 0]  # CLS token

        features = self.backbone(x)

        # Some models (e.g. newer DINO checkpoints) return a dict
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            elif "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1)
            else:
                features = next(iter(features.values()))
                if features.ndim > 2:
                    features = features.mean(dim=1)
                return features

        if isinstance(features, torch.Tensor) and features.ndim > 2:
            if hasattr(self.backbone, "cls_token"):
                return features[:, 0]  # CLS token
            return features.mean(dim=1)  # global average pool

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def forward_head(self, projected_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(projected_features)

    def _use_cached_features(self) -> bool:
        """Check if we're being fed cached features (set by training script)."""
        return getattr(self, "_cached_mode", False)

    def _calculate_loss(self, batch: tuple, mode: str = "train") -> torch.Tensor:
        inputs, labels = batch
        logits = (
            self.forward_head(inputs) if self._use_cached_features() else self(inputs)
        )

        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = (
            self.forward_head(inputs) if self._use_cached_features() else self(inputs)
        )

        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        preds = logits.argmax(dim=1)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_per_class_acc(preds, labels)

        return loss

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        per_class_acc = self.val_per_class_acc.compute()

        self.log("val_f1_macro", f1, prog_bar=True)
        self.log("val_precision_macro", precision, prog_bar=True)
        self.log("val_recall_macro", recall, prog_bar=True)

        for i in range(self.num_classes):
            self.log(f"val_acc_class_{i}", per_class_acc[i])

        self.log("val_acc_std", torch.std(per_class_acc), prog_bar=True)
        self.log("val_acc_min", torch.min(per_class_acc), prog_bar=True)

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_per_class_acc.reset()

    def configure_optimizers(self):
        if self.backbone_is_frozen:
            # SGD + cosine LR per facebookresearch/dino/eval_linear.py.
            # Linear LR scaling (Goyal et al., 2017 — "Accurate, Large Minibatch SGD").
            scaled_lr = float(self.hparams.lr) * float(self.hparams.batch_size) / 256.0

            optimizer = optim.SGD(
                self.classifier.parameters(),
                lr=scaled_lr,
                momentum=0.9,
                weight_decay=float(self.hparams.weight_decay),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=0
            )
        else:
            # Differential LR — DINOv2 issue #276 recommends much lower backbone LR
            # to avoid feature collapse.
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            head_params = list(self.classifier.parameters())

            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(self.hparams.lr) * 0.01},
                    {"params": head_params},
                ],
                lr=float(self.hparams.lr),
                weight_decay=float(self.hparams.weight_decay),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=0
            )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser) -> Any:
        parser = parent_parser.add_argument_group("BaseClassifier")
        parser.add_argument(
            "--backbone",
            type=str,
            default="resnet18",
            help="Backbone architecture",
        )
        parser.add_argument(
            "--frozen",
            type=bool,
            default=False,
            help="Whether to freeze backbone layers",
        )
        parser.add_argument(
            "--freeze_strategy",
            type=str,
            default="all",
            choices=["all", "last_layer"],
            help="How to freeze the backbone when --frozen is set",
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            default=10,
            help="Number of output classes",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Base learning rate (scaled by batch_size/256)",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay (default 0.0 for linear eval)",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Batch size (used for linear LR scaling rule)",
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            help="Maximum number of training epochs",
        )
        parser.add_argument(
            "--use_gradient_checkpointing",
            type=bool,
            default=False,
            help="Whether to use gradient checkpointing to save memory",
        )
        return parent_parser
