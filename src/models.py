"""ResNet18 with a shared convolutional backbone and two dataset-specific heads."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = ROOT / "checkpoints" / "pretrained"
RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_FILE = PRETRAINED_DIR / "resnet18-f37072fd.pth"
# Keep torch's cache inside the project: the user profile is read-only in the
# sandboxed execution environment.
os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints" / ".torch_cache"))


def _resnet18_imagenet() -> nn.Module:
    """Load torchvision's ImageNet ResNet18 weights from a local file.

    The weights are downloaded once to ``checkpoints/pretrained`` so that the
    experiment is reproducible without relying on the user's torch cache.
    """
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    if RESNET18_FILE.exists():
        state = torch.load(RESNET18_FILE, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(
            RESNET18_URL, model_dir=str(PRETRAINED_DIR), map_location="cpu",
            check_hash=True, progress=True,
        )
        torch.save(state, RESNET18_FILE)
    net = resnet18(weights=None)
    net.load_state_dict(state)
    return net


@dataclass
class ClassifierOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    def __getitem__(self, index):
        # Hugging Face Trainer accepts tuple-like outputs as well as
        # ModelOutput objects; this keeps the dependency surface small.
        if isinstance(index, slice):
            return (self.loss, self.logits)[index]
        if index == 0:
            return self.loss
        if index == 1:
            return self.logits
        raise IndexError(index)

    def __len__(self) -> int:
        return 2


class DualHeadResNet18(nn.Module):
    """One ImageNet-pretrained backbone, one 10-way and one 100-way head.

    ``task_ids`` selects which head is used, so the same class can serve both
    CIFAR-10 and CIFAR-100 training/evaluation without changing the Trainer.
    """

    def __init__(self, num_classes_10: int = 10, num_classes_100: int = 100,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        backbone = _resnet18_imagenet() if pretrained else resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout)
        self.fc_cifar10 = nn.Linear(in_features, num_classes_10)
        self.fc_cifar100 = nn.Linear(in_features, num_classes_100)
        self.num_classes_10 = num_classes_10
        self.num_classes_100 = num_classes_100

    def forward(self, pixel_values: torch.Tensor,
                task_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> ClassifierOutput:
        features = self.backbone(pixel_values)
        features = self.dropout(features)

        if task_ids is None:
            task_id = 0
        elif isinstance(task_ids, torch.Tensor):
            task_id = int(task_ids.reshape(-1)[0].item())
            if bool((task_ids != task_id).any()):
                raise ValueError("A batch must contain a single task_id")
        else:
            task_id = int(task_ids)

        if task_id == 0:
            logits = self.fc_cifar10(features)
        elif task_id == 1:
            logits = self.fc_cifar100(features)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.long())
        return ClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


class FeatureBackbone(nn.Module):
    """Frozen ImageNet ResNet18 backbone used for the main clustering experiment."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = _resnet18_imagenet() if pretrained else resnet18(weights=None)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = 512

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values)


def build_model(pretrained: bool = True, device: str | torch.device = "cpu") -> DualHeadResNet18:
    model = DualHeadResNet18(pretrained=pretrained)
    return model.to(device)


def load_checkpoint(path, device: str | torch.device = "cpu") -> DualHeadResNet18:
    model = DualHeadResNet18(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable),
            "total_million": round(total / 1e6, 2)}


def architecture_table(model: DualHeadResNet18) -> list[dict]:
    rows = [
        {"stage": "输入", "module": "预处理", "output": "3×H×W (H=W=输入尺寸)"},
        {"stage": "Stem", "module": "7×7 Conv, stride 2 + BN + ReLU + 3×3 MaxPool", "output": "64×H/4×W/4"},
        {"stage": "Layer 1", "module": "2 × BasicBlock, 64 channels", "output": "64×H/4×W/4"},
        {"stage": "Layer 2", "module": "2 × BasicBlock, 128 channels, stride 2", "output": "128×H/8×W/8"},
        {"stage": "Layer 3", "module": "2 × BasicBlock, 256 channels, stride 2", "output": "256×H/16×W/16"},
        {"stage": "Layer 4", "module": "2 × BasicBlock, 512 channels, stride 2", "output": "512×H/32×W/32"},
        {"stage": "Pooling", "module": "Global Average Pooling", "output": "512"},
        {"stage": "CIFAR-10 头", "module": "Dropout(0.2) + Linear(512→10)", "output": "10"},
        {"stage": "CIFAR-100 头", "module": "Dropout(0.2) + Linear(512→100)", "output": "100"},
    ]
    return rows
