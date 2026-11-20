"""CIFAR-10 data loading with a fixed train/validation/test partition."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config


def get_transforms(augment: bool = False):
    """Return the image transform; augmentation is for training images only."""
    normalize = transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD)
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),     # small translations
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),            # small rotations
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])


def get_split_indices():
    """Fixed-seed split of the 50,000 training images into train / validation."""
    generator = torch.Generator().manual_seed(config.SEED)
    indices = torch.randperm(config.NUM_TRAIN + config.NUM_VAL, generator=generator).tolist()
    return indices[:config.NUM_TRAIN], indices[config.NUM_TRAIN:]


def get_loaders(augment: bool = False):
    """Return (train, train_eval, val, test) DataLoaders.

    train      : training images (augmented if augment=True), shuffled
    train_eval : the same training images, never augmented (for training accuracy)
    val        : 5,000 held-out training images, never augmented
    test       : official 10,000 test images; final evaluation only
    """
    plain_tf = get_transforms(augment=False)
    train_tf = get_transforms(augment=augment)

    train_full = datasets.CIFAR10(config.DATA_DIR, train=True, download=True, transform=train_tf)
    train_full_plain = datasets.CIFAR10(config.DATA_DIR, train=True, download=True, transform=plain_tf)
    test_set = datasets.CIFAR10(config.DATA_DIR, train=False, download=True, transform=plain_tf)

    train_idx, val_idx = get_split_indices()
    train_set = Subset(train_full, train_idx)
    train_eval_set = Subset(train_full_plain, train_idx)
    val_set = Subset(train_full_plain, val_idx)

    loader_args = dict(batch_size=config.BATCH_SIZE,
                       num_workers=config.NUM_WORKERS,
                       pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(train_eval_set, shuffle=False, **loader_args),
        DataLoader(val_set, shuffle=False, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )