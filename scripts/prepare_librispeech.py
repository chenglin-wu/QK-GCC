# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from qkgcc.simulation import (
    TEST_ROOM,
    TRAIN_ROOM,
    remove_silence,
    save_frame,
    simulate_pair,
)


FS = 16000
FRAME_LENGTH = 2048
MAX_TAU = 23
SNR_GRID = (0, 6, 12, 18, 24, 30)
T60_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
VAL_SPEAKERS = {260, 672, 908}
TEST_SPEAKERS = {61, 121, 237}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the LibriSpeech TDE dataset")
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def waveform(dataset: torchaudio.datasets.LIBRISPEECH, index: int) -> tuple[torch.Tensor, int]:
    audio, sample_rate, _, speaker_id, _, _ = dataset[index]
    if sample_rate != FS:
        audio = torchaudio.functional.resample(audio, sample_rate, FS)
    return remove_silence(audio), speaker_id


def ensure_empty(directory: Path) -> None:
    if directory.exists() and next(directory.glob("*.pt"), None):
        raise FileExistsError(f"output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def generate_random_split(
    dataset: torchaudio.datasets.LIBRISPEECH,
    indices: list[int],
    locations: np.ndarray,
    output: Path,
    acoustic_conditions: int,
    windows: int,
    flip: bool,
    rng: np.random.Generator,
) -> None:
    ensure_empty(output)
    counter = 0
    for index in tqdm(indices, desc=output.name):
        signal, speaker_id = waveform(dataset, index)
        for _ in range(acoustic_conditions):
            snr = rng.uniform(0.0, 30.0)
            t60 = rng.uniform(0.2, 1.0)
            x1, x2, delay = simulate_pair(signal, locations[index], TRAIN_ROOM, FS, snr, t60)
            max_start = min(len(signal), len(x1), len(x2)) - FRAME_LENGTH
            for _ in range(windows):
                if max_start <= 0:
                    continue
                start = int(rng.integers(0, max_start))
                stop = start + FRAME_LENGTH
                save_frame(
                    output / f"sample_{counter}.pt",
                    x1[start:stop], x2[start:stop], delay, snr, t60, str(speaker_id),
                )
                counter += 1
                if flip:
                    save_frame(
                        output / f"sample_{counter}.pt",
                        x2[start:stop], x1[start:stop], -delay, snr, t60, str(speaker_id),
                    )
                    counter += 1
    print(f"{output.name}: {counter} pairs")


def generate_test(
    dataset: torchaudio.datasets.LIBRISPEECH,
    indices: list[int],
    locations: np.ndarray,
    output: Path,
) -> None:
    for snr in SNR_GRID:
        for t60 in T60_GRID:
            directory = output / "test" / f"snr_{snr}_t60_{t60}"
            ensure_empty(directory)
            counter = 0
            for index in tqdm(indices, desc=f"SNR={snr}, T60={t60}"):
                signal, speaker_id = waveform(dataset, index)
                np.random.seed(index)
                x1, x2, delay = simulate_pair(signal, locations[index], TEST_ROOM, FS, snr, t60)
                for window in range(15):
                    start, stop = window * FRAME_LENGTH, (window + 1) * FRAME_LENGTH
                    if stop > min(len(x1), len(x2)):
                        break
                    save_frame(
                        directory / f"sample_{counter}.pt",
                        x1[start:stop], x2[start:stop], delay, snr, t60, str(speaker_id),
                    )
                    counter += 1
            print(f"SNR={snr}, T60={t60}: {counter} pairs")


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=args.librispeech_root, url="test-clean", download=args.download
    )
    train_locations = np.random.uniform(
        np.zeros(3), TRAIN_ROOM.dimensions, (len(dataset), 3)
    )
    val_locations = np.random.uniform(np.zeros(3), TRAIN_ROOM.dimensions, (len(dataset), 3))
    test_locations = np.random.uniform(np.zeros(3), TEST_ROOM.dimensions, (len(dataset), 3))
    valid, speakers = [], {}
    for index in tqdm(range(len(dataset)), desc="scan"):
        signal, speaker_id = waveform(dataset, index)
        if len(signal) >= 2 * FS:
            valid.append(index)
            speakers[index] = speaker_id
    train = [i for i in valid if speakers[i] not in VAL_SPEAKERS | TEST_SPEAKERS]
    val = [i for i in valid if speakers[i] in VAL_SPEAKERS]
    test = [i for i in valid if speakers[i] in TEST_SPEAKERS]
    rng = np.random.default_rng(args.seed)
    generate_random_split(dataset, train, train_locations, args.output_root / "train", 3, 10, True, rng)
    generate_random_split(dataset, val, val_locations, args.output_root / "val", 2, 5, False, rng)
    generate_test(dataset, test, test_locations, args.output_root)


if __name__ == "__main__":
    main()
