"""Eksperimen 3: Progressive unfreezing dengan differential learning rate.

Usage:
    python train_exp3_progressive.py --data "Q:/Pembelajaran Mendalam/Studio 2/train" --epochs 15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    if train_dir.exists() or validation_dir.exists():
        raise RuntimeError(f"Split tidak lengkap di {split_dir}. Hapus folder tersebut lalu ulangi.")

    rng = random.Random(seed)
    for class_name in CLASSES:
        source_class = source_dir / class_name
        if not source_class.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {source_class}")
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
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1_values) / num_classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)
    return macro_f1(targets, predictions, len(CLASSES)), accuracy


def load_pretrained_backbone(model: nn.Module) -> None:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    pretrained = models.resnet18(weights=weights).state_dict()
    pretrained = {k: v for k, v in pretrained.items() if not k.startswith("fc.")}
    model.load_state_dict(pretrained, strict=False)


def set_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--work-dir", type=Path, default=Path("local_data"))
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--unfreeze-epoch", type=int, default=5, help="Epoch untuk unfreeze backbone")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    train_dir, validation_dir = make_local_split(
        args.data.resolve(), args.work_dir.resolve(), args.validation_fraction, args.seed
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=get_transform())

    counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(CLASSES)).float()
    class_weights = (counts.sum() / (len(CLASSES) * counts)).to(device)
    print(f"class_counts={counts.tolist()}  class_weights={class_weights.tolist()}")

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = get_model()
    load_pretrained_backbone(model)
    model.to(device)

    # Fase 1: Freeze backbone, train head only
    set_trainable(model, False)
    for param in model.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.unfreeze_epoch)

    submission_dir = Path("submission")
    submission_dir.mkdir(exist_ok=True)
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    print(f"device={device}  train={len(train_dataset)}  val={len(validation_dataset)}")
    print("=== FASE 1: Training head only ===")

    for epoch in range(1, args.unfreeze_epoch + 1):
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), submission_dir / "model.pth")
            print(f"saved=submission/model.pth best_macro_f1={best_macro_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"no_improve={epochs_no_improve}/{args.patience}")

    # Fase 2: Unfreeze all dengan differential LR
    print("=== FASE 2: Full fine-tuning ===")
    set_trainable(model, True)

    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": args.lr_head},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": args.lr_backbone},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.unfreeze_epoch)
    epochs_no_improve = 0

    for epoch in range(args.unfreeze_epoch + 1, args.epochs + 1):
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), submission_dir / "model.pth")
            print(f"saved=submission/model.pth best_macro_f1={best_macro_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"no_improve={epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print(f"early_stopping at epoch {epoch}")
                break

    print(f"best_macro_f1={best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
