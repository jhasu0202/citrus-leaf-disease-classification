from pathlib import Path

import torch
from torch import nn
from torchvision import models
import timm

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5

MODEL_CONFIGS = {
    "ResNet34": {
        "type": "resnet",
        "path": "models/ResNet34/resnet_best_model.pt",
    },
    "EfficientNet-B0": {
        "type": "efficientnet",
        "path": "models/EfficientNet_B0/efficientnet_best_model.pt",
    },
    "ViT-B/16": {
        "type": "vit",
        "path": "models/MobileNetV3_Large/vit_best_model.pt",
    },
    "MobileNetV3-Large": {
        "type": "mobilenet",
        "path": "models/MobileNetV3_Large/mobilenet_best_model.pt",
    },
    "ConvNeXt-Tiny": {
        "type": "convnext",
        "path": "models/ConvNeXt_Tiny/convnext_best_model.pt",
    },
    "Swin-Tiny": {
        "type": "swin",
        "path": "models/Swin_Tiny/swin_best_model.pt",
    },
    "MobileViT-Small": {
        "type": "mobilevit",
        "path": "models/MobileViT_Small/mobilevit_best_model.pt",
    },
}

CLASS_NAMES = [
    "Black spot",
    "Canker",
    "Greening",
    "Healthy",
    "Melanose",
]

_model_cache = {}


def create_architecture(model_type):
    if model_type == "resnet":
        model = models.resnet34(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )

    elif model_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
        )

    elif model_type == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, NUM_CLASSES
        )

    elif model_type == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, NUM_CLASSES
        )

    elif model_type == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )

    elif model_type == "swin":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    elif model_type == "mobilevit":
        model = timm.create_model(
            "mobilevit_s",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )

    else:
        raise ValueError(f"Unsupported architecture: {model_type}")

    return model


def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    checkpoint_path = ROOT / config["path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state_dict.items()
    }

    model = create_architecture(config["type"])
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def get_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)

    return _model_cache[model_name]