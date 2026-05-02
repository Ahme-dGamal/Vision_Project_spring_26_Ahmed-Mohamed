"""
finetune_shvit_food101.py

Wrapper that fine-tunes SHViT on Food-101 with a single GPU.

Design choices that match the original SHViT paper / engine.py:
  - Augmentation: RandAugment (rand-m9-mstd0.5-inc1) + RandomErasing (p=0.25)
                  + Mixup (alpha=0.8) + CutMix (alpha=1.0) + label smoothing 0.1
  - Forward pass during training runs in full FP32 (matching the commented-out
    autocast in the original engine.py's train_one_epoch).
  - Eval forward pass uses torch.cuda.amp.autocast (matching original evaluate()).
  - GradScaler is kept for gradient overflow protection.
  - Gradient clipping via AGC-style norm clip (clip_grad=0.02, matching paper).
  - Head keys dropped by shape mismatch (same logic as main.py --finetune path).
  - LR is used directly — no linear world-size scaling since world_size=1.

Usage:
    python finetune_shvit_food101.py \\
        --shvit-dir  /path/to/SHViT \\
        --finetune   /path/to/shvit_s4.pth \\
        --data-root  /path/to/data \\
        --output-dir /path/to/output \\
        --epochs 30

Outputs:
    <output-dir>/training_log.csv        one row per epoch
    <output-dir>/best.pth                highest val top-1 (model state_dict only)
    <output-dir>/checkpoint_<N>.pth      resumable checkpoint every --save-freq epochs
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as tvdatasets
from timm.data import Mixup, create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import accuracy


NUM_CLASSES = 101


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("SHViT Food-101 fine-tuning (single GPU)")

    # Paths
    p.add_argument("--shvit-dir",   type=Path, default=Path("SHViT"))
    p.add_argument("--finetune",    type=Path, default=Path("weights/shvit_s4.pth"),
                   help="ImageNet pretrained checkpoint to start from")
    p.add_argument("--data-root",   type=Path, default=Path("data"))
    p.add_argument("--output-dir",  type=Path, default=Path("checkpoints/shvit_food101"))
    p.add_argument("--resume",      type=Path, default=None,
                   help="Resume a previous fine-tuning run from a checkpoint_N.pth")

    # Model
    p.add_argument("--model", default="SHViT_S4",
                   choices=["SHViT_S1", "SHViT_S2", "SHViT_S3", "SHViT_S4"])
    p.add_argument("--input-size", default=224, type=int)

    # Training
    p.add_argument("--epochs",        default=30,   type=int)
    p.add_argument("--batch-size",    default=64,   type=int)
    p.add_argument("--lr",            default=1e-4, type=float,
                   help="Peak LR (no world-size scaling; paper used 1e-3 / 8 GPUs / bs-256)")
    p.add_argument("--min-lr",        default=1e-6, type=float)
    p.add_argument("--warmup-epochs", default=5,    type=int)
    p.add_argument("--weight-decay",  default=0.025, type=float)
    p.add_argument("--clip-grad",     default=0.02,  type=float,
                   help="Gradient norm clip threshold (0 = disabled)")
    p.add_argument("--seed",          default=0, type=int)
    p.add_argument("--num-workers",   default=2, type=int)
    p.add_argument("--save-freq",     default=10, type=int,
                   help="Save a resumable checkpoint every N epochs")

    # Augmentation — same defaults as SHViT paper
    p.add_argument("--aa",        default="rand-m9-mstd0.5-inc1", type=str,
                   help="AutoAugment policy (timm format)")
    p.add_argument("--smoothing", default=0.1,     type=float)
    p.add_argument("--reprob",    default=0.25,    type=float,
                   help="Random Erasing probability")
    p.add_argument("--remode",    default="pixel", type=str)
    p.add_argument("--mixup",     default=0.8,     type=float)
    p.add_argument("--cutmix",    default=1.0,     type=float)

    # Eval-only mode
    p.add_argument("--eval", action="store_true",
                   help="Run evaluation only (requires --resume or --finetune with matching classes)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transform(args, is_train: bool):
    if is_train:
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=1,
        )
    # Val: resize to 256, centre-crop to 224 (timm default for is_training=False)
    return create_transform(
        input_size=args.input_size,
        is_training=False,
        interpolation="bicubic",
    )


def build_loaders(args):
    train_ds = tvdatasets.Food101(
        root=str(args.data_root), split="train",
        transform=build_transform(args, is_train=True), download=True,
    )
    val_ds = tvdatasets.Food101(
        root=str(args.data_root), split="test",
        transform=build_transform(args, is_train=False), download=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=int(1.5 * args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Food-101  train: {len(train_ds):,}  val: {len(val_ds):,}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model + checkpoint helpers
# ---------------------------------------------------------------------------

def load_pretrained(model: torch.nn.Module, ckpt_path: Path) -> None:
    """
    Load ImageNet weights, dropping head keys whose shape mismatches.
    Mirrors the --finetune branch in SHViT's main.py exactly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    # SHViT classifier: head.l.{weight,bias}  (distillation variant: head_dist.l.*)
    head_keys = [
        "head.l.weight", "head.l.bias",
        "head_dist.l.weight", "head_dist.l.bias",
    ]
    model_sd = model.state_dict()
    removed = []
    for k in head_keys:
        if k in state_dict and state_dict[k].shape != model_sd.get(k, torch.empty(0)).shape:
            del state_dict[k]
            removed.append(k)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights loaded from {ckpt_path}")
    print(f"  Removed (shape mismatch) : {removed}")
    if msg.missing_keys:
        print(f"  Missing keys           : {msg.missing_keys}")


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, args) -> float:
    """Linear warmup then cosine decay, matching timm's cosine scheduler."""
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, args, mixup_fn):
    """
    Full-FP32 forward pass (matching original SHViT engine.py which has
    `if True:  # with torch.cuda.amp.autocast():` — autocast intentionally off).
    GradScaler is kept for gradient overflow protection during backprop.
    """
    model.train()
    total_loss, n = 0.0, 0

    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * samples.size(0)
        n += samples.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    """AMP autocast during eval — matching original engine.py's evaluate()."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # timm accuracy() returns values in [0, 100]
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bs = images.size(0)
        total_loss += loss.item() * bs
        top1_sum   += acc1.item() * bs
        top5_sum   += acc5.item() * bs
        n += bs

    return total_loss / n, top1_sum / n / 100.0, top5_sum / n / 100.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Register SHViT model family with timm
    sys.path.insert(0, str(args.shvit_dir.resolve()))
    try:
        import models  # noqa: F401 — side-effect: registers SHViT_* with timm
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import SHViT models from {args.shvit_dir}.\n"
            f"Make sure --shvit-dir points to the cloned SHViT repo.\n{exc}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    print(f"Model: {args.model}   Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = args.output_dir / "training_log.csv"
    best_path = args.output_dir / "best.pth"

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader = build_loaders(args)

    # ---- Model --------------------------------------------------------------
    model = create_model(args.model, pretrained=False, num_classes=NUM_CLASSES)
    if args.finetune and args.finetune.exists():
        load_pretrained(model, args.finetune)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Loss / Mixup -------------------------------------------------------
    mixup_active = args.mixup > 0 or args.cutmix > 0
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=1.0, switch_prob=0.5, mode="batch",
            label_smoothing=args.smoothing, num_classes=NUM_CLASSES,
        )
    criterion = (
        SoftTargetCrossEntropy()
        if mixup_active
        else LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    )

    # ---- Optimizer / scaler -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ---- Resume -------------------------------------------------------------
    start_epoch = 0
    best_top1   = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_top1   = ckpt.get("best_top1", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  (best top-1: {best_top1*100:.2f}%)")

    # ---- Eval only ----------------------------------------------------------
    if args.eval:
        _, top1, top5 = evaluate_epoch(model, val_loader, device)
        print(f"Eval  top-1: {top1*100:.2f}%   top-5: {top5*100:.2f}%")
        return

    # ---- CSV header (only on fresh run) ------------------------------------
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "val_loss", "val_top1", "val_top5", "time_sec"]
            )

    # ---- Training loop ------------------------------------------------------
    print(f"\nFine-tuning for {args.epochs} epochs (starting at {start_epoch})")
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(epoch, args)
        set_lr(optimizer, lr)

        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, args, mixup_fn,
        )
        val_loss, val_top1, val_top5 = evaluate_epoch(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        print(
            f"[{epoch+1:3d}/{args.epochs}] lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"top1={val_top1*100:.2f}%  top5={val_top5*100:.2f}%  ({elapsed:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{lr:.6e}",
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{val_top1:.4f}", f"{val_top5:.4f}",
                f"{elapsed:.1f}",
            ])

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_top1": val_top1, "val_top5": val_top5}, best_path)
            print(f"  -> new best ({val_top1*100:.2f}%), saved to {best_path}")

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            ckpt_path = args.output_dir / f"checkpoint_{epoch+1:03d}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_top1": best_top1,
                "args": vars(args),
            }, ckpt_path)

    print(f"\nDone.  Best val top-1: {best_top1*100:.2f}%")
    print(f"Log  : {csv_path}")
    print(f"Best : {best_path}")


if __name__ == "__main__":
    main()
