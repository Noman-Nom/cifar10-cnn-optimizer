import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar10_loaders(batch_size, data_dir='./data', num_workers=4, val_split=0.1):
    """
    Create DataLoader objects for CIFAR-10 with train/val/test splits.

    Args:
        batch_size (int): Batch size for the dataloaders.
        data_dir (str): Directory to download/load CIFAR-10 data.
        num_workers (int): Number of worker processes for data loading.
        val_split (float): Fraction of training data to use for validation.

    Returns:
        (train_loader, val_loader, test_loader): DataLoader objects.
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transforms
    )

    n_total = len(full_train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = list(range(n_total))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
