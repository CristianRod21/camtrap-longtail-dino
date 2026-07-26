from typing import Literal

import torch.nn as nn


def freeze_model_layers(
    model: nn.Module,
    frozen: bool,
    strategy: Literal["all", "last_layer"] = "all",
) -> nn.Module:
    """
    Freeze model layers based on the specified strategy.

    Args:
        model: The backbone model to freeze.
        frozen: Whether to freeze at all. If False, returns model unchanged.
        strategy: Freezing strategy.
            - "all": Freeze the entire backbone.
            - "last_layer": Freeze everything except the last block/layer
              so it can still be fine-tuned.

    Returns:
        The model with the appropriate layers frozen.

    Raises:
        ValueError: If the strategy is not supported.
    """
    if not frozen:
        return model

    if strategy not in ("all", "last_layer"):
        raise ValueError(
            f"Unknown freeze strategy '{strategy}'. "
            f"Supported strategies: 'all', 'last_layer'"
        )

    class_name = model.__class__.__name__.lower()

    if "resnet" in class_name:
        return _freeze_resnet(model, strategy)
    elif "convnext" in class_name:
        return _freeze_convnext(model, strategy)
    elif "efficientnet" in class_name:
        return _freeze_efficientnet(model, strategy)
    elif "swin" in class_name:
        return _freeze_swin(model, strategy)
    elif "visiontransformer" in class_name:
        return _freeze_vit(model, strategy)
    elif "dino" in class_name:
        return _freeze_dino(model, strategy)
    else:
        # Generic fallback: freeze everything for "all",
        # warn and freeze everything for "last_layer" since we
        # don't know the architecture layout.
        if strategy == "last_layer":
            import warnings

            warnings.warn(
                f"Model '{model.__class__.__name__}' is not explicitly supported "
                f"for 'last_layer' freezing. Falling back to freezing all layers. "
                f"Consider adding explicit support for this architecture.",
                stacklevel=2,
            )
        _freeze_all(model)
        return model


def _freeze_all(model: nn.Module) -> None:
    """Freeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze(module: nn.Module) -> None:
    """Unfreeze every parameter in a module."""
    for param in module.parameters():
        param.requires_grad = True


# Architecture-specific implementations. All follow the same pattern: freeze
# everything first, then selectively unfreeze the last block + head for
# "last_layer" strategy.


def _freeze_resnet(model: nn.Module, strategy: str) -> nn.Module:
    """
    ResNet: layers are conv1, bn1, layer1-4, fc.
    "last_layer" keeps layer4 and fc trainable.
    """
    _freeze_all(model)

    if strategy == "last_layer":
        if hasattr(model, "layer4"):
            _unfreeze(model.layer4)
        if hasattr(model, "fc"):
            _unfreeze(model.fc)

    return model


def _freeze_convnext(model: nn.Module, strategy: str) -> nn.Module:
    """
    ConvNeXt: features is a Sequential of 8 stages (pairs of downsampling + blocks).
    "last_layer" keeps the last stage and classifier trainable.
    """
    _freeze_all(model)

    if strategy == "last_layer":
        if hasattr(model, "features") and len(model.features) > 0:
            _unfreeze(model.features[-1])
        if hasattr(model, "classifier"):
            _unfreeze(model.classifier)

    return model


def _freeze_efficientnet(model: nn.Module, strategy: str) -> nn.Module:
    """
    EfficientNet V2: features is a Sequential of MBConv blocks.
    "last_layer" keeps the last feature block and classifier trainable.
    """
    _freeze_all(model)

    if strategy == "last_layer":
        if hasattr(model, "features") and len(model.features) > 0:
            _unfreeze(model.features[-1])
        if hasattr(model, "classifier"):
            _unfreeze(model.classifier)

    return model


def _freeze_swin(model: nn.Module, strategy: str) -> nn.Module:
    """
    Swin Transformer: features contains patch embedding + sequential stages.
    "last_layer" keeps the last stage and head trainable.
    """
    _freeze_all(model)

    if strategy == "last_layer":
        if hasattr(model, "features") and len(model.features) > 0:
            _unfreeze(model.features[-1])
        if hasattr(model, "head"):
            _unfreeze(model.head)

    return model


def _freeze_vit(model: nn.Module, strategy: str) -> nn.Module:
    _freeze_all(model)

    if strategy == "last_layer":
        # timm ViTs use model.blocks
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            _unfreeze(model.blocks[-1])
        # torchvision ViTs use model.encoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            _unfreeze(model.encoder.layers[-1])

        if hasattr(model, "head"):
            _unfreeze(model.head)
        elif hasattr(model, "heads"):
            _unfreeze(model.heads)

        # timm has a final norm before the head
        if hasattr(model, "norm") or hasattr(model, "fc_norm"):
            _unfreeze(getattr(model, "norm", None) or model.fc_norm)

    return model


def _freeze_dino(model: nn.Module, strategy: str) -> nn.Module:
    """
    DINOv2 / DINOv3: uses model.blocks (a ModuleList of transformer blocks).
    "last_layer" keeps the last block, final norm, and head trainable.
    """
    _freeze_all(model)

    if strategy == "last_layer":
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            _unfreeze(model.blocks[-1])
        # DINOv2 has a final LayerNorm that should stay trainable
        if hasattr(model, "norm"):
            _unfreeze(model.norm)
        if hasattr(model, "head"):
            _unfreeze(model.head)

    return model
