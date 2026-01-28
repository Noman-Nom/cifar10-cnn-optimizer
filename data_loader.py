import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=64, augment=True):
    """
    Create DataLoader objects for CIFAR-10 training and testing datasets.
    
    Args:
        batch_size (int): Batch size for the dataloaders.
        augment (bool): Whether to apply data augmentation on training data.
    
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing datasets.
    """
    # CIFAR-10 dataset normalization values (mean and std for RGB channels)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    # Data augmentation and normalization for training
    if augment:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    # Normalization for testing (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example: load with default parameters and print batch shape
    train_loader, test_loader = get_cifar10_loaders()
    for images, labels in train_loader:
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        break
