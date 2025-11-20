#!/usr/bin/env python3
"""
MNIST Speed-Run — Per-Batch Full Eval (Data-Only Levels) — v7 (PyTorch)

What this does
--------------
- Trains a simple CNN or Logistic Regression on MNIST.
- After **every training batch**, runs a **full validation pass** (test split).
- Uses **exactly 30 thresholds** for validation CE (nats), evenly spaced from **2.3 → 1.0**.
- When the current val CE drops below the next threshold, it **unlocks that level** and writes:
    levels/L{idx}_{threshold}/
      ├── metrics.json             (no titles/insights; just numbers)
      └── confusion_matrix.json    (10×10 integers)
- By default, **exits early** once all 30 levels are captured.
  Use `--keep-training-after-complete` to continue training anyway.
- Final run summary saved at the run root: `metrics.json` and the final `confusion_matrix.json`.

Notes
-----
- Loss is in **nats** (PyTorch CrossEntropyLoss uses ln).
- This is compute-heavy (full eval per batch). Reduce epochs or batch size on CPU.
"""

import argparse
import os
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# Thresholds (30 evenly spaced in [2.3, 1.0], desc)
# -----------------------------


def build_thresholds() -> List[float]:
    vals = np.linspace(2.3, 1.0, 30).tolist()
    # sort descending (high→low) for unlocking order
    return sorted([float(v) for v in vals], reverse=True)


# -----------------------------
# Models (NO dropout anywhere)
# -----------------------------


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 28→14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Data
# -----------------------------


def make_loaders(batch_size: int, data_root: str, seed: int, workers: int = 2):
    g = torch.Generator().manual_seed(seed)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test, batch_size=1024, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, test_loader


# -----------------------------
# Evaluation
# -----------------------------


@torch.no_grad()
def evaluate_full(model, loader, device) -> Dict[str, Any]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    cm = np.zeros((10, 10), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        total += y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        for yt, pt in zip(y.view(-1).tolist(), pred.view(-1).tolist()):
            cm[yt, pt] += 1
    return {"val_ce": loss_sum / total, "val_acc": correct / total, "cm": cm.tolist()}


# -----------------------------
# I/O helpers
# -----------------------------


def write_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# Training loop (per-batch eval, data-only levels)
# -----------------------------


def train(args):
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, test_loader = make_loaders(
        args.batch_size, args.data_root, args.seed, args.num_workers
    )

    model = LogisticRegression() if args.model == "logreg" else SimpleCNN()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    thresholds = build_thresholds()  # 30 values, high→low
    next_idx = 0  # next threshold index to unlock

    # Output structure
    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(
        args.outdir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    )
    os.makedirs(run_dir, exist_ok=True)
    levels_dir = os.path.join(run_dir, "levels")
    os.makedirs(levels_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "thresholds.json"), {"thresholds": thresholds})

    per_step = []
    global_step = 0

    print(
        f"[run] device={device} model={args.model} epochs={args.epochs} batch={args.batch_size} lr={args.lr} amp={args.amp}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_sum = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, reduction="mean")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_sum += loss.item() * yb.size(0)
            seen += yb.size(0)
            global_step += 1

            # Full validation after each batch
            stats = evaluate_full(model, test_loader, device)
            train_ce_running = running_sum / max(seen, 1)
            per_step.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_ce_running": train_ce_running,
                    "val_ce": stats["val_ce"],
                    "val_acc": stats["val_acc"],
                }
            )
            print(
                f"[eval@step {global_step}] val_ce={stats['val_ce']:.4f}  val_acc={stats['val_acc']*100:.2f}%"
            )

            # Unlock next threshold if reached
            if next_idx < len(thresholds) and stats["val_ce"] <= thresholds[next_idx]:
                th = thresholds[next_idx]
                lvl_dir = os.path.join(levels_dir, f"L{next_idx+1:02d}_{th:.3f}")
                os.makedirs(lvl_dir, exist_ok=True)
                # Write metrics.json (data only)
                write_json(
                    os.path.join(lvl_dir, "metrics.json"),
                    {
                        "level_index": next_idx + 1,
                        "threshold": th,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_ce_running": train_ce_running,
                        "val_ce": stats["val_ce"],
                        "val_acc": stats["val_acc"],
                        "timestamp": time.time(),
                    },
                )
                # Write confusion matrix as JSON
                write_json(
                    os.path.join(lvl_dir, "confusion_matrix.json"),
                    {"matrix": stats["cm"]},
                )
                print(
                    f"[level] unlocked L{next_idx+1:02d} at threshold ≤ {th:.3f} (step {global_step})"
                )
                next_idx += 1

                # Early exit if all 30 levels captured
                if (
                    next_idx >= len(thresholds)
                    and not args.keep_training_after_complete
                ):
                    print("[levels] All 30 thresholds captured — exiting early.")
                    final = stats
                    # Final summary
                    write_json(
                        os.path.join(run_dir, "metrics.json"),
                        {
                            "model": args.model,
                            "device": str(device),
                            "epochs_seen": epoch,
                            "batch_size": args.batch_size,
                            "lr": args.lr,
                            "amp": args.amp,
                            "final": {
                                "val_ce": final["val_ce"],
                                "val_acc": final["val_acc"],
                            },
                            "levels_captured": next_idx,
                            "thresholds": thresholds,
                            "per_step": per_step,
                        },
                    )
                    # Also dump final confusion matrix JSON
                    write_json(
                        os.path.join(run_dir, "confusion_matrix.json"),
                        {"matrix": final["cm"]},
                    )
                    print("[done] artifacts saved to:", run_dir)
                    return

        # End-of-epoch progress line
        print(
            f"[epoch {epoch}] avg_train_ce={running_sum/max(seen,1):.4f}  levels_captured={next_idx}/30"
        )

    # If not early-exited, write final summary at the end
    final = evaluate_full(model, test_loader, device)
    write_json(
        os.path.join(run_dir, "metrics.json"),
        {
            "model": args.model,
            "device": str(device),
            "epochs_seen": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp": args.amp,
            "final": {"val_ce": final["val_ce"], "val_acc": final["val_acc"]},
            "levels_captured": next_idx,
            "thresholds": thresholds,
            "per_step": per_step,
        },
    )
    write_json(os.path.join(run_dir, "confusion_matrix.json"), {"matrix": final["cm"]})
    print(
        "[final] val_ce={:.4f}  val_acc={:.2f}%".format(
            final["val_ce"], final["val_acc"] * 100.0
        )
    )
    print("[done] artifacts saved to:", run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "cnn"], default="cnn")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--amp", action="store_true", help="Enable mixed precision (CUDA only)"
    )
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument(
        "--keep-training-after-complete",
        action="store_true",
        help="Do NOT exit early when all 30 levels are captured",
    )
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
