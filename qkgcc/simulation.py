# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
import torch
from librosa.effects import split


@dataclass(frozen=True)
class RoomConfig:
    dimensions: tuple[float, float, float]
    microphones: np.ndarray


TRAIN_ROOM = RoomConfig(
    (7.0, 5.0, 3.0),
    np.array([[3.5, 2.25, 1.5], [3.5, 2.75, 1.5]], dtype=np.float64).T,
)
TEST_ROOM = RoomConfig(
    (6.0, 4.0, 2.5),
    np.array([[3.0, 1.75, 1.25], [3.0, 2.25, 1.25]], dtype=np.float64).T,
)


def remove_silence(signal: torch.Tensor, top_db: float = 20.0) -> torch.Tensor:
    signal = signal.squeeze()
    intervals = split(signal.numpy(), top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return signal[:0]
    return torch.cat([signal[start:end] for start, end in intervals])


def source_positions(count: int, room: RoomConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(np.zeros(3), np.asarray(room.dimensions), size=(count, 3))


def simulate_pair(
    signal: np.ndarray | torch.Tensor,
    source: np.ndarray,
    room_config: RoomConfig,
    sample_rate: int,
    snr: float,
    t60: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    absorption, max_order = pra.inverse_sabine(t60, room_config.dimensions)
    room = pra.ShoeBox(
        room_config.dimensions,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    signal = np.asarray(signal, dtype=np.float32).squeeze()
    room.add_source(source, signal=signal)
    room.add_microphone(room_config.microphones)
    distances = np.linalg.norm(room_config.microphones.T - source, axis=1)
    delay = (distances[0] - distances[1]) * sample_rate / room.c
    room.simulate(reference_mic=0, snr=snr)
    return room.mic_array.signals[0], room.mic_array.signals[1], float(delay)


def save_frame(
    path: str,
    x1: np.ndarray,
    x2: np.ndarray,
    delay: float,
    snr: float,
    t60: float,
    source_id: str,
    max_tau: int = 23,
) -> None:
    torch.save(
        {
            "x1": torch.as_tensor(x1, dtype=torch.float32),
            "x2": torch.as_tensor(x2, dtype=torch.float32),
            "delay": float(delay + max_tau),
            "snr": float(snr),
            "t60": float(t60),
            "source_id": source_id,
        },
        path,
    )

