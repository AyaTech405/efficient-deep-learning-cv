"""Model evaluation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .data import create_dataloaders, set_seed
from .models import build_model, model_summary


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate a classifier and return standard classification metrics."""
    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def evaluate_checkpoint(
    model_name: str,
    checkpoint: str,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Load a checkpoint and evaluate it."""
    model = build_model(model_name, pretrained=False).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)

    metrics = evaluate_model(model, loader, device)
    parameters, size_mb = model_summary(model)
    metrics.update({"parameters": parameters, "model_size_mb": size_mb})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on CIFAR-10.")
    parser.add_argument("--model", choices=["resnet18", "mobilenetv3-small"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/evaluation.json")
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
    metrics = evaluate_checkpoint(args.model, args.checkpoint, test_loader, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
