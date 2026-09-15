# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import math

import torch


class TDEMetrics:
    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.n = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.correct = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        error = logits.argmax(-1) - target
        self.n += target.numel()
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.float().square().sum().item()
        self.correct += (error.abs() < self.tolerance).sum().item()

    def result(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mae": self.absolute_error / self.n,
            "rmse": math.sqrt(self.squared_error / self.n),
            "acc_10cm": self.correct / self.n,
        }

