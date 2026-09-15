# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qkgcc import QKGCC
from qkgcc.data import OfflineTDOADataset
from qkgcc.metrics import TDEMetrics


CONDITION = re.compile(r"snr_(-?\d+)_t60_([0-9.]+)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QK-GCC")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> QKGCC:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = QKGCC(**config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.eval()


def conditions(root: Path) -> list[tuple[int, float, Path]]:
    test_root = root / "test" if (root / "test").is_dir() else root
    found = []
    for path in test_root.iterdir():
        match = CONDITION.fullmatch(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), float(match.group(2)), path))
    return sorted(found, key=lambda item: (item[0], item[1]))


def main() -> None:
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tolerance = 10 * 16000 / (343 * 100)
    total = TDEMetrics(tolerance)
    rows = []
    with torch.inference_mode():
        for snr, t60, directory in conditions(args.data_root):
            dataset = OfflineTDOADataset(directory)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            meter = TDEMetrics(tolerance)
            for x1, x2, target in tqdm(loader, desc=f"SNR={snr}, T60={t60}", leave=False):
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                target = target.long().to(device, non_blocking=True)
                logits = model(x1, x2)
                meter.update(logits, target)
                total.update(logits, target)
            row = {"snr": snr, "t60": t60, **meter.result()}
            rows.append(row)
            print(
                f"SNR={snr:>2}, T60={t60:.1f}: MAE={row['mae']:.4f}, "
                f"RMSE={row['rmse']:.4f}, Acc@10cm={100 * row['acc_10cm']:.2f}%"
            )

    if not rows:
        raise FileNotFoundError(f"no test conditions found under {args.data_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = total.result()
    print(
        f"TOTAL n={result['n']} MAE={result['mae']:.4f} "
        f"RMSE={result['rmse']:.4f} Acc@10cm={100 * result['acc_10cm']:.2f}%"
    )


if __name__ == "__main__":
    main()

