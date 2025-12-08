from typing import Dict, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from loss import ClipLoss

class ResidualAdd(nn.Module):
    """Wraps a module with a residual connection."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class EEGProjectLayer(nn.Module):
    """
    Lightweight encoder that flattens (channels, timesteps) EEG blocks and
    projects them into a z_dim embedding using a residual MLP head.
    """

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int], drop_proj: float = 0.3):
        super().__init__()
        self.c_num = c_num
        self.timesteps = timesteps
        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(z_dim, z_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(z_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self.loss_func = ClipLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_dim)
        return self.model(x)


class FlattenHead(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().view(x.size(0), -1)


class BaseModel(nn.Module):
    """
    Shared backbone holder for convolutional EEG encoders.
    Sub-classes fill self.backbone with a nn.Sequential stack.
    """

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int], embedding_dim: int = 1440):
        super().__init__()
        self.backbone: nn.Module | None = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(z_dim, z_dim),
                    nn.Dropout(0.5),
                )
            ),
            nn.LayerNorm(z_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            raise RuntimeError("EEG backbone is not defined.")
        x = x.unsqueeze(1)
        x = self.backbone(x)
        return self.project(x)


class Shallownet(BaseModel):
    """Two-layer temporal convolution with channel mixing and pooling."""

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int]):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )


class Deepnet(BaseModel):
    """Stacked temporal convolutions with progressively larger channel width."""

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int]):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1400)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )


class EEGnet(BaseModel):
    """Compact architecture using separable temporal convolutions."""

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int]):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1248)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.5),
        )


class TSconv(BaseModel):
    """Temporal convolution front-end followed by spatial mixing."""

    def __init__(self, z_dim: int, c_num: int, timesteps: Tuple[int, int]):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )


EEG_ENCODERS: Dict[str, Type[nn.Module]] = {
    "EEGProjectLayer": EEGProjectLayer,
    "Shallownet": Shallownet,
    "Deepnet": Deepnet,
    "EEGnet": EEGnet,
    "TSconv": TSconv,
}


def build_eeg_encoder(name: str, *args, **kwargs) -> nn.Module:
    """Factory helper to keep downstream code clean."""
    if name not in EEG_ENCODERS:
        raise KeyError(f"Unknown EEG encoder '{name}'. Valid keys: {list(EEG_ENCODERS)}")
    return EEG_ENCODERS[name](*args, **kwargs)


__all__ = [
    "ResidualAdd",
    "EEGProjectLayer",
    "FlattenHead",
    "BaseModel",
    "Shallownet",
    "Deepnet",
    "EEGnet",
    "TSconv",
    "EEG_ENCODERS",
    "build_eeg_encoder",
]
