from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AnalyticNetAN001BnReLU(nn.Module):
    """Baseline model: BatchNorm plus ReLU encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h


class AnalyticNetAN002NoBnTanh(nn.Module):
    """Variant without BatchNorm, using Tanh activations."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h


class AnalyticNetAN003Linear(nn.Module):
    """Variant without explicit nonlinear activation functions."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h


class AnalyticNetAN004DeepTanh(nn.Module):
    """Variant with one extra hidden layer and Tanh activations."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h


MODEL_REGISTRY = {
    "AN001_BN_RELU": AnalyticNetAN001BnReLU,
    "AN002_NO_BN_TANH": AnalyticNetAN002NoBnTanh,
    "AN003_LINEAR": AnalyticNetAN003Linear,
    "AN004_DEEP_TANH": AnalyticNetAN004DeepTanh,
}


def build_model(
    model_id: str,
    input_dim: int,
    hidden_dim: int,
    bottleneck_dim: int,
) -> nn.Module:
    """Instantiate a model variant from its stable identifier."""

    try:
        model_cls = MODEL_REGISTRY[model_id]
    except KeyError as exc:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unsupported MODEL_ID '{model_id}'. Expected one of: {supported}.") from exc
    return model_cls(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
