# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineTDOADataset(Dataset):
    def __init__(self, directory: str | Path, sig_len: int = 2048, max_tau: int = 23) -> None:
        self.files = sorted(Path(directory).glob("*.pt"))
        self.sig_len = sig_len
        self.max_tau = max_tau

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = torch.load(self.files[index], map_location="cpu", weights_only=False)
        x1, x2 = sample["x1"].float(), sample["x2"].float()
        target = int(round(float(sample["delay"])))
        if x1.shape != (self.sig_len,) or x2.shape != (self.sig_len,):
            raise ValueError(f"invalid waveform shape in {self.files[index]}")
        if not 0 <= target <= 2 * self.max_tau:
            raise ValueError(f"invalid delay target in {self.files[index]}")
        return x1.unsqueeze(0), x2.unsqueeze(0), target


class PackedTDOADataset(Dataset):
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.signals = np.load(directory / "signals.npy", mmap_mode="r")
        self.targets = np.load(directory / "targets.npy", mmap_mode="r")
        expected = (meta["count"], 2, meta["sig_len"])
        if self.signals.shape != expected or self.targets.shape != (meta["count"],):
            raise ValueError(f"invalid packed dataset at {directory}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pair = torch.from_numpy(np.array(self.signals[index], copy=True))
        return pair[0:1], pair[1:2], int(self.targets[index])


def swap_channels(
    x1: torch.Tensor,
    x2: torch.Tensor,
    target: torch.Tensor,
    max_tau: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.rand(target.shape[0], device=target.device) < probability
    view = mask.reshape((target.shape[0],) + (1,) * (x1.ndim - 1))
    return (
        torch.where(view, x2, x1),
        torch.where(view, x1, x2),
        torch.where(mask, 2 * max_tau - target, target),
    )

