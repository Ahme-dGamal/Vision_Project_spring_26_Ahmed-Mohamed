"""
train_baseline.py
Fine-tune ResNet-50 or MobileNetV2 on Food-101 as a baseline for SHViT.

Usage:
    python train_baseline.py --model resnet50      --data-root data --output-dir checkpoints
    python train_baseline.py --model mobilenet_v2  --data-root data --output-dir checkpoints

Outputs (per model):
    <output-dir>/<model>/training_log.csv   one row per epoch
    <output-dir>/<model>/best.pth           checkpoint of highest val top-1
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)


NUM_CLASSES = 101
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_dataloaders(data_root: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = torchvision.datasets.Food101(
        root=data_root, split="train", transform=train_tf, download=True,
    )
    val_ds = torchvision.datasets.Food101(
        root=data_root, split="test", transform=val_tf, download=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_model(name: str) -> nn.Module:
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)

        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(targets.view(-1, 1).expand_as(pred5))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += images.size(0)
    return total_loss / n, top1 / n, top5 / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}   Device: {device}   Epochs: {args.epochs}")

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_log.csv"
    best_path = out_dir / "best.pth"

    train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss", "val_top1", "val_top5", "time_sec"]
        )

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"lr={lr:.2e}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  top1={val_top1*100:.2f}%  "
            f"top5={val_top5*100:.2f}%  ({elapsed:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}",
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{val_top1:.4f}", f"{val_top5:.4f}",
                f"{elapsed:.1f}",
            ])

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({
                "model_name": args.model,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
            }, best_path)
            print(f"  -> new best, saved to {best_path}")

    print(f"\nDone. Best val top-1: {best_top1*100:.2f}%")
    print(f"Log:  {csv_path}")
    print(f"Ckpt: {best_path}")


if __name__ == "__main__":
    main()
