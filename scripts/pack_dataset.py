# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


NUMBER = re.compile(r"(\d+)$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack .pt pairs into memory-mapped arrays")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def key(path: Path) -> int:
    match = NUMBER.search(path.stem)
    return int(match.group(1)) if match else 0


def pack(source: Path, destination: Path) -> None:
    files = sorted(source.glob("*.pt"), key=key)
    if not files:
        raise FileNotFoundError(f"no .pt files found in {source}")
    first = torch.load(files[0], map_location="cpu", weights_only=False)
    sig_len = int(first["x1"].numel())
    destination.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        destination / "signals.npy", mode="w+", dtype=np.float32, shape=(len(files), 2, sig_len)
    )
    targets = np.lib.format.open_memmap(
        destination / "targets.npy", mode="w+", dtype=np.int16, shape=(len(files),)
    )
    for index, path in enumerate(tqdm(files, desc=source.name)):
        sample = torch.load(path, map_location="cpu", weights_only=False)
        signals[index, 0] = sample["x1"].numpy()
        signals[index, 1] = sample["x2"].numpy()
        targets[index] = round(float(sample["delay"]))
    signals.flush()
    targets.flush()
    metadata = {"count": len(files), "sig_len": sig_len, "max_tau": 23}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    for split in ("train", "val"):
        pack(args.input_root / split, args.output_root / split)


if __name__ == "__main__":
    main()

