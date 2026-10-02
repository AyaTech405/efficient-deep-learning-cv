"""Benchmark accuracy, size and inference latency for trained models."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .data import create_dataloaders, set_seed
from .evaluate import evaluate_checkpoint
from .models import build_model, model_summary


def load_model(model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    """Load a trained model checkpoint."""
    model = build_model(model_name, pretrained=False).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    return model.eval()


def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 5,
    iterations: int = 20,
) -> float:
    """Measure mean batch inference latency in milliseconds."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative.")
    sample = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return ((time.perf_counter() - start) / iterations) * 1000


def benchmark_model(
    model_name: str,
    checkpoint: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a checkpoint and measure model size and latency."""
    model = load_model(model_name, checkpoint, device)
    metrics = evaluate_checkpoint(model_name, checkpoint, test_loader, device)
    parameters, size_mb = model_summary(model)
    latency_ms = measure_latency(model, device)
    metrics.update({
        "model": model_name,
        "parameters": parameters,
        "model_size_mb": size_mb,
        "inference_latency_ms": latency_ms,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trained models.")
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--mobilenet-checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_size=1,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    rows = [
        benchmark_model("resnet18", args.resnet_checkpoint, test_loader, device),
        benchmark_model("mobilenetv3-small", args.mobilenet_checkpoint, test_loader, device),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark results to {output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
