"""Eksperimen 1: ResNet-18 + Weighted Loss + Augmentasi Agresif + Early Stopping

Strategi:
- WeightedRandomSampler untuk oversampling kelas minoritas (kertas_kardus)
- Weighted CrossEntropyLoss
- Augmentasi agresif: rotation, grayscale, stronger color jitter
- Early stopping patience=5 berdasarkan Macro-F1
- Batch size 64 optimal untuk GPU T4

Usage:
    python train_exp1_weighted.py --data /path/to/train --epochs 15 --batch-size 64
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from submission.model import get_model, get_transform

CLASSES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_local_split(
    source_dir: Path,
    split_dir: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    train_dir = split_dir / "my_train"
    validation_dir = split_dir / "my_val"
    if train_dir.exists() and validation_dir.exists():
        return train_dir, validation_dir

    rng = random.Random(seed)
    for class_name in CLASSES:
        source_class = source_dir / class_name
        files = sorted(source_class.glob("*.jpg"))
        rng.shuffle(files)
        validation_count = int(len(files) * validation_fraction)

        for index, source_file in enumerate(files):
            target_root = validation_dir if index < validation_count else train_dir
            target_file = target_root / class_name / source_file.name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

    return train_dir, validation_dir


def macro_f1(targets: list[int], predictions: list[int], num_classes: int) -> float:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    f1_values = []
    for class_index in range(num_classes):
        tp = confusion[class_index, class_index].item()
        fp = confusion[:, class_index].sum().item() - tp
        fn = confusion[class_index, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1_values) / num_classes


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            targets.extend(labels.tolist())

    accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)
    return macro_f1(targets, predictions, len(CLASSES)), accuracy


def load_pretrained_backbone(model: nn.Module) -> None:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    pretrained = models.resnet18(weights=weights).state_dict()
    pretrained = {k: v for k, v in pretrained.items() if not k.startswith("fc.")}
    model.load_state_dict(pretrained, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Folder train dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--work-dir", type=Path, default=Path("local_data"))
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs harus minimal 1")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction harus di antara 0 dan 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, validation_dir = make_local_split(
        args.data.resolve(), args.work_dir.resolve(), args.validation_fraction, args.seed
    )

    # Augmentasi agresif untuk training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_dataset = datasets.ImageFolder(validation_dir, transform=get_transform())
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    if train_dataset.classes != CLASSES or validation_dataset.classes != CLASSES:
        raise RuntimeError(
            f"Urutan kelas tidak sesuai. train={train_dataset.classes}, expected={CLASSES}"
        )

    # WeightedRandomSampler untuk oversampling kelas minoritas
    targets_tensor = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets_tensor, minlength=len(CLASSES)).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets_tensor]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 2, "pin_memory": True}
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = get_model()
    load_pretrained_backbone(model)
    model.to(device)

    # Weighted loss
    loss_weight = (class_counts.sum() / (len(CLASSES) * class_counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    submission_dir = Path("submission")
    submission_dir.mkdir(exist_ok=True)
    best_macro_f1 = -1.0
    patience_counter = 0
    print(f"device={device}  train={len(train_dataset)}  val={len(validation_dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()

        current_macro_f1, current_accuracy = evaluate(model, validation_loader, device)
        average_loss = total_loss / len(train_dataset)
        print(
            f"epoch={epoch:02d} loss={average_loss:.4f} "
            f"val_macro_f1={current_macro_f1:.4f} val_accuracy={current_accuracy:.4f}"
        )

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            torch.save(model.state_dict(), submission_dir / "model.pth")
            print(f"saved=submission/model.pth best_macro_f1={best_macro_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining selesai. Best Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
