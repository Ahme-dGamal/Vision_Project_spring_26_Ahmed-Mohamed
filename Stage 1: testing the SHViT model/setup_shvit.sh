#!/usr/bin/env bash
# setup_shvit.sh — Full SHViT setup for food classification
# Run from: /home/user/Vision_Project_spring_26
set -e

PROJECT_DIR="$(pwd)"
SHVIT_DIR="$PROJECT_DIR/SHViT"
WEIGHTS_DIR="$PROJECT_DIR/weights"
DATA_DIR="$PROJECT_DIR/data"

echo "=== Step 1: Clone SHViT ==="
if [ ! -d "$SHVIT_DIR" ]; then
    git clone https://github.com/ysj9909/SHViT.git "$SHVIT_DIR"
else
    echo "SHViT already cloned, skipping."
fi

echo ""
echo "=== Step 2: Create conda environment ==="
conda create -y -n shvit python=3.9
# Activate in your shell with: conda activate shvit
# All pip commands below assume the shvit env is active.

echo ""
echo "=== Step 3: Install PyTorch 1.11 + requirements ==="
# PyTorch 1.11 with CUDA 11.3 — adjust cu113 to cpu if no GPU
conda run -n shvit pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

conda run -n shvit pip install -r "$SHVIT_DIR/requirements.txt"

echo ""
echo "=== Step 4: Download SHViT-S4 pretrained weights ==="
mkdir -p "$WEIGHTS_DIR"
wget -c "https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth" \
    -O "$WEIGHTS_DIR/shvit_s4.pth"
echo "Weights saved to $WEIGHTS_DIR/shvit_s4.pth"

echo ""
echo "=== Done! ==="
echo "Next steps:"
echo "  conda activate shvit"
echo "  python prepare_food101.py          # download + organize Food-101"
echo "  python verify_model.py             # quick eval on 50 images"
