"""CIFAR-10 data loading with deterministic train/validation separation."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_loaders(
    root: Path,
    batch_size: int = 128,
    validation_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    augment = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    standardize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train_augmented = datasets.CIFAR10(
        root, train=True, download=download, transform=augment
    )
    train_evaluation = datasets.CIFAR10(
        root, train=True, download=False, transform=standardize
    )
    test_set = datasets.CIFAR10(
        root, train=False, download=download, transform=standardize
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_augmented), generator=generator).tolist()
    validation_size = int(len(permutation) * validation_fraction)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]

    train_loader = DataLoader(
        Subset(train_augmented, training_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        Subset(train_evaluation, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, validation_loader, test_loader, train_augmented.classes
