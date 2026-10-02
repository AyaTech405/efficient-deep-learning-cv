"""CIFAR-10 data loading utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 224


def set_seed(seed: int) -> None:
    """Set seeds for reproducible Python, NumPy and PyTorch runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_transform(train: bool) -> transforms.Compose:
    """Build the ImageNet-compatible transform used by pretrained models."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def _subset_dataset(
    dataset: Dataset,
    subset_size: Optional[int],
    seed: int,
) -> Dataset:
    """Return a reproducible random subset when subset_size is specified."""
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    if subset_size <= 0:
        raise ValueError("subset_size must be positive or None.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())


def create_dataloaders(
    data_dir: str | Path = "data",
    train_size: Optional[int] = 2000,
    test_size: Optional[int] = 500,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and create train/test data loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=_build_transform(train=True),
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=_build_transform(train=False),
    )

    train_dataset = _subset_dataset(train_dataset, train_size, seed)
    test_dataset = _subset_dataset(test_dataset, test_size, seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
