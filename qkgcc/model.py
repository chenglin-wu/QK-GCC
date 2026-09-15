# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyPosition(nn.Module):
    def __init__(self, n_freq: int, d_model: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_freq, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class DelayHead(nn.Module):
    def __init__(
        self,
        n_heads: int,
        band_width: int,
        max_tau: int,
        hidden_channels: int = 32,
        reliability_mix: float = 0.9,
    ) -> None:
        super().__init__()
        self.band_width = band_width
        channels = n_heads * (2 * band_width + 1)
        groups = math.gcd(8, hidden_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 5, padding=2),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.freq_gate = nn.Conv1d(hidden_channels, 1, 1)
        nn.init.zeros_(self.freq_gate.weight)
        nn.init.zeros_(self.freq_gate.bias)
        mix_logit = math.log(reliability_mix / (1.0 - reliability_mix))
        self.reliability_mix_logit = nn.Parameter(torch.tensor(mix_logit))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2 * max_tau + 1),
        )

    def _decompose(self, corr: torch.Tensor) -> torch.Tensor:
        b = self.band_width
        center = corr[:, :, b:b + 1]
        positive = corr[:, :, b + 1:]
        negative = corr[:, :, :b].flip(2)
        symmetric = 0.5 * (positive + negative)
        antisymmetric = 0.5 * (positive - negative)
        return torch.cat((center, symmetric, antisymmetric), dim=2)

    def forward(self, corr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._decompose(corr)
        batch, heads, offsets, n_freq = x.shape
        x = self.conv(x.reshape(batch, heads * offsets, n_freq))
        learned = torch.softmax(self.freq_gate(x), dim=-1)
        mix = torch.sigmoid(self.reliability_mix_logit)
        reliability = mix * learned + (1.0 - mix) / n_freq
        pooled = (x * reliability).sum(dim=-1)
        return self.mlp(pooled), reliability


class QKGCC(nn.Module):
    """Single-pass local Query-Key spectral matcher for TDE."""

    def __init__(
        self,
        sig_len: int = 2048,
        max_tau: int = 23,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        band_width: int = 8,
        eps: float = 1e-6,
        reliability_mix: float = 0.9,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.sig_len = sig_len
        self.n_fft = 2 * sig_len
        self.n_freq = self.n_fft // 2 + 1
        self.max_tau = max_tau
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.band_width = band_width
        self.eps = eps

        self.input_proj = nn.Linear(3, d_model)
        self.freq_pos = FrequencyPosition(self.n_freq, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.delay_head = DelayHead(
            n_heads, band_width, max_tau, reliability_mix=reliability_mix
        )

    def _tokens(self, signal: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(signal, n=self.n_fft)
        magnitude = spectrum.abs()
        safe = magnitude.clamp_min(self.eps)
        features = torch.stack(
            (torch.log1p(magnitude), spectrum.real / safe, spectrum.imag / safe),
            dim=-1,
        )
        return self.pre_norm(self.freq_pos(self.input_proj(features)))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_freq, _ = x.shape
        return x.reshape(batch, n_freq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _local_match(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        batch, heads, n_freq, head_dim = query.shape
        b = self.band_width
        key = F.pad(key, (0, 0, b, b))
        key = key.reshape(batch * heads, n_freq + 2 * b, head_dim).permute(0, 2, 1)
        windows = key.unfold(2, 2 * b + 1, 1)
        query = query.reshape(batch * heads, n_freq, head_dim).permute(0, 2, 1).unsqueeze(-1)
        scores = (query * windows).sum(1).reshape(batch, heads, n_freq, 2 * b + 1)
        scale = head_dim**-0.5 * self.logit_scale.clamp(-4.0, 4.0).exp()
        return (scores * scale).permute(0, 1, 3, 2)

    @staticmethod
    def _prepare(signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim == 3 and signal.shape[1] == 1:
            signal = signal[:, 0]
        if signal.ndim != 2:
            raise ValueError(f"expected [B,T] or [B,1,T], got {tuple(signal.shape)}")
        return signal

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1, x2 = self._prepare(x1), self._prepare(x2)
        if x1.shape != x2.shape or x1.shape[-1] != self.sig_len:
            raise ValueError("input pair has an invalid shape")
        q1 = self._heads(self.W_q(self._tokens(x1)))
        k2 = self._heads(self.W_k(self._tokens(x2)))
        corr = self.dropout(self._local_match(q1, k2))
        logits, reliability = self.delay_head(corr)
        if not return_aux:
            return logits
        return logits, {
            "correlation": corr,
            "reliability": reliability,
            "reliability_mix": torch.sigmoid(self.delay_head.reliability_mix_logit),
        }

    def predict(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self(x1, x2).argmax(dim=-1) - self.max_tau

