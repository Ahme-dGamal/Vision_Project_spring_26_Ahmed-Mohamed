"""
prepare_food101.py
Downloads Food-101 via torchvision and reorganizes it into the
ImageNet-style layout that SHViT's main.py expects:

  data/food101/
    train/
      apple_pie/
        img1.jpg
        ...
      baby_back_ribs/
        ...
    val/
      apple_pie/
        ...

Usage:
    python prepare_food101.py [--root data/food101]
"""

import argparse
import shutil
from pathlib import Path

import torchvision.datasets as datasets


def download_food101(root: Path) -> None:
    """Download both splits via torchvision (handles the tar automatically)."""
    print(f"Downloading Food-101 train split to {root} ...")
    datasets.Food101(root=str(root), split="train", download=True)
    print(f"Downloading Food-101 test split to {root} ...")
    datasets.Food101(root=str(root), split="test", download=True)


def reorganize(root: Path) -> None:
    """
    torchvision unpacks Food-101 as:
        <root>/food-101/images/<class>/<id>.jpg
        <root>/food-101/meta/train.txt   (lines: class/id)
        <root>/food-101/meta/test.txt

    We read the official split files and hard-link (or copy) images
    into train/ and val/ subdirectories.
    """
    food_root = root / "food-101"
    images_dir = food_root / "images"
    meta_dir = food_root / "meta"

    split_map = {"train": "train.txt", "val": "test.txt"}

    for split_name, meta_file in split_map.items():
        split_dir = root / split_name
        meta_path = meta_dir / meta_file

        print(f"\nOrganizing {split_name} split ...")
        lines = meta_path.read_text().strip().splitlines()

        for line in lines:
            class_name, img_id = line.split("/")
            src = images_dir / class_name / f"{img_id}.jpg"
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{img_id}.jpg"

            if not dst.exists():
                # Use hard link to avoid doubling disk usage; fall back to copy
                try:
                    dst.hardlink_to(src)
                except (AttributeError, OSError):
                    shutil.copy2(src, dst)

        n_classes = len(list(split_dir.iterdir()))
        n_images = sum(1 for _ in split_dir.rglob("*.jpg"))
        print(f"  {split_name}: {n_classes} classes, {n_images} images -> {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and organize Food-101")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/food101"),
        help="Directory to store the dataset (default: data/food101)",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)

    download_food101(args.root)
    reorganize(args.root)

    print("\nDone! Dataset layout:")
    print(f"  {args.root}/train/<class>/<image>.jpg")
    print(f"  {args.root}/val/<class>/<image>.jpg")
    print("\nPass to SHViT main.py with: --data-path", args.root)


if __name__ == "__main__":
    main()
