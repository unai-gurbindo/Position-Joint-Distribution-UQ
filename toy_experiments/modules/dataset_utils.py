import numpy as np
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def cifar10_loaders(config, data_transformation="transf_data"):
    # Data transformations
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
    if data_transformation == "raw_data":
        print(f"Using raw data for CIFAR-10")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif data_transformation == "transf_data":
        print(f"Using transformed data transformation for CIFAR-10")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    # Load CIFAR-10 dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


######## OOD #########
class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        self.data = np.load(
            os.path.join(root, f"{corruption}.npy")
        )[ (severity - 1) * 10000 : severity * 10000 ]
        self.targets = np.load(
            os.path.join(root, "labels.npy")
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def load_svhn_dataset(root, transform):
    return datasets.SVHN(
        root=root,
        split='test',
        download=True,
        transform=transform
    )

class DatasetWithOODFlag(torch.utils.data.Dataset):
    def __init__(self, dataset, is_ood: bool):
        self.dataset = dataset
        self.is_ood = is_ood

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, int(self.is_ood)
