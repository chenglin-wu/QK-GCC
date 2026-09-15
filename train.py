# Copyright (c) 2026 jinkz33@XMU&UofG

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from qkgcc import QKGCC
from qkgcc.data import PackedTDOADataset, swap_channels
from qkgcc.metrics import TDEMetrics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QK-GCC")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qkgcc"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--swap-probability", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: QKGCC,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    swap_probability: float,
    use_amp: bool,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metrics = TDEMetrics(10 * 16000 / (343 * 100))
    total_loss, started = 0.0, time.perf_counter()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for x1, x2, target in tqdm(loader, leave=False):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            target = target.long().to(device, non_blocking=True)
            if training:
                x1, x2, target = swap_channels(x1, x2, target, 23, swap_probability)
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x1, x2)
                loss = F.cross_entropy(logits, target)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * target.numel()
            metrics.update(logits.detach(), target)
    result = metrics.result()
    result.update(loss=total_loss / result["n"], seconds=time.perf_counter() - started)
    return result


def main() -> None:
    args = arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PackedTDOADataset(args.data_root / "train")
    val_set = PackedTDOADataset(args.data_root / "val")
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    model = QKGCC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, best_mae = 0, float("inf")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_mae = state["best_val_mae"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(start_epoch, args.epochs):
        generator.manual_seed(args.seed + epoch)
        train_stats = run_epoch(
            model, train_loader, device, optimizer, scaler, args.swap_probability, use_amp
        )
        val_stats = run_epoch(model, val_loader, device, None, scaler, 0.0, use_amp)
        scheduler.step()
        best_mae = min(best_mae, float(val_stats["mae"]))
        record = {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_mae,
            "model_config": {
                "sig_len": 2048,
                "max_tau": 23,
                "d_model": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "band_width": 8,
            },
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if float(val_stats["mae"]) == best_mae:
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(
            f"epoch={epoch + 1:02d} train_mae={train_stats['mae']:.4f} "
            f"val_mae={val_stats['mae']:.4f} val_rmse={val_stats['rmse']:.4f} "
            f"val_acc={100 * val_stats['acc_10cm']:.2f}%"
        )


if __name__ == "__main__":
    main()
