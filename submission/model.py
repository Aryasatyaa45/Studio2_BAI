"""Arsitektur submission Studio 2."""

import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model() -> nn.Module:
    """Bangun arsitektur tanpa mengunduh bobot atau memuat model.pth."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform() -> transforms.Compose:
    """Transform validasi resmi untuk gambar RGB."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
