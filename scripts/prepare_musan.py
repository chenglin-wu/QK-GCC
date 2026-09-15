# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from qkgcc.simulation import TEST_ROOM, save_frame, simulate_pair, source_positions


FS = 16000
FRAME_LENGTH = 2048
SNR_GRID = (0, 6, 12, 18, 24, 30)
T60_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate MUSAN zero-shot TDE data")
    parser.add_argument("--musan-root", type=Path, required=True)
    parser.add_argument("--source-type", choices=("music", "noise"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-files", type=int)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--windows", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_chunks(
    root: Path, source_type: str, num_files: int, seconds: float, rng: np.random.Generator
) -> list[tuple[np.ndarray, str]]:
    files = sorted((root / source_type).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no WAV files found under {root / source_type}")
    if len(files) > num_files:
        chosen = sorted(rng.choice(len(files), num_files, replace=False).tolist())
        files = [files[index] for index in chosen]
    length = int(FS * seconds)
    threshold = 0.01 if source_type == "music" else 0.005
    chunks = []
    for path in tqdm(files, desc=f"load {source_type}"):
        audio, sample_rate = torchaudio.load(path)
        audio = audio.mean(0)
        if sample_rate != FS:
            audio = torchaudio.functional.resample(audio, sample_rate, FS)
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
        for start in range(0, len(audio) - length + 1, length):
            chunk = audio[start:start + length]
            if chunk.square().mean().sqrt() >= threshold:
                chunks.append((chunk.numpy(), path.stem))
    if not chunks:
        raise RuntimeError("no non-silent chunks were found")
    return chunks


def main() -> None:
    args = arguments()
    rng = np.random.default_rng(args.seed)
    default_files = 5 if args.source_type == "music" else 50
    chunks = load_chunks(
        args.musan_root,
        args.source_type,
        args.num_files or default_files,
        args.chunk_seconds,
        rng,
    )
    locations = source_positions(len(chunks), TEST_ROOM, rng)
    for snr in SNR_GRID:
        for t60 in T60_GRID:
            directory = args.output_root / "test" / f"snr_{snr}_t60_{t60}"
            if directory.exists() and next(directory.glob("*.pt"), None):
                raise FileExistsError(f"output directory is not empty: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            counter = 0
            for index, (signal, source_id) in enumerate(tqdm(chunks, desc=f"SNR={snr}, T60={t60}")):
                np.random.seed(index)
                x1, x2, delay = simulate_pair(signal, locations[index], TEST_ROOM, FS, snr, t60)
                for window in range(args.windows):
                    start, stop = window * FRAME_LENGTH, (window + 1) * FRAME_LENGTH
                    if stop > min(len(x1), len(x2)):
                        break
                    save_frame(
                        directory / f"sample_{counter}.pt",
                        x1[start:stop], x2[start:stop], delay, snr, t60, source_id,
                    )
                    counter += 1
            print(f"SNR={snr}, T60={t60}: {counter} pairs")


if __name__ == "__main__":
    main()
