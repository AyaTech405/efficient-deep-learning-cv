"""Training entry point for CIFAR-10 image classification."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW

from .data import create_dataloaders, set_seed
from .models import build_model
from .evaluate import evaluate_model

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return total_loss / total_items


def save_history(history: list[Dict[str, float]], path: Path) -> None:
    """Save epoch-level metrics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 image classifier.")
    parser.add_argument("--model", choices=["resnet18", "mobilenetv3-small"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--history", default="results/training_history.csv")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[Dict[str, float]] = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_model(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(row)
        LOGGER.info(
            "Epoch %d/%d | loss=%.4f | accuracy=%.4f | f1=%.4f",
            epoch, args.epochs, loss, metrics["accuracy"], metrics["f1"],
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = output_dir / f"{args.model.replace('-', '_')}_best.pt"
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                checkpoint,
            )
            LOGGER.info("Saved best checkpoint: %s", checkpoint)

    save_history(history, Path(args.history))
    LOGGER.info("Training history saved to %s", args.history)


if __name__ == "__main__":
    main()
