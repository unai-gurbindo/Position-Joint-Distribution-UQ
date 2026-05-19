import numpy as np
import torch
import torch.nn.functional as F


def train_model(model, loader, optimizer, DEVICE):
    """
    Function to train an individual CNN.
    """
    model.train()

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
