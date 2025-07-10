#!/usr/bin/env python3
"""
train_yolo_filtered.py
----------------------
Train an Ultralytics-YOLO detection model on the **filtered** dataset produced by
`megadetector_frame_reprocessor.py`.

Assumptions
===========
1. Dataset folder follows standard YOLO directory layout::

       <dataset_root>/
           dataset.yaml            # already present (if missing we create it)
           train/<species>/*.jpg
           val/<species>/*.jpg
           test/<species>/*.jpg

2. Ultralytics-YOLO (>=8.1) is installed (already in requirements.txt).

Quick start (train a small model for 100 epochs)::

    python train_yolo_filtered.py \
            --data_dir Kaggle/cupybara/yolo11_dataset_filtered \
            --model yolov8n.pt \
            --epochs 100 \
            --batch 32 \
            --imgsz 640

The script will:
1. Auto-generate `dataset.yaml` if not found (based on sub-folders under *train/*).
2. Launch training with `ultralytics.YOLO`.
3. Save runs to *runs/filtered_train/* (can be customised).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import os
from typing import Optional

# --------------------------------------------------------------------------------------
# Dataset utilities
# --------------------------------------------------------------------------------------

def discover_classes(train_dir: Path):
    """Return list of class names based on sub-folder names under *train*."""
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def create_dataset_yaml(dataset_root: Path):
    """Create a minimal dataset.yaml if it doesn't exist."""
    yaml_path = dataset_root / "dataset.yaml"
    if yaml_path.exists():
        print(f"📄 Using existing dataset.yaml → {yaml_path}")
        return yaml_path

    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"train directory not found under {dataset_root}")

    classes = discover_classes(train_dir)
    cfg = {
        "path": str(dataset_root.resolve()).replace("\\", "/"),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": classes,
        "nc": len(classes),
    }
    with yaml_path.open("w") as f:
        yaml.dump(cfg, f)
    print(f"📝 Created dataset.yaml with {len(classes)} classes at {yaml_path}")
    return yaml_path

# --------------------------------------------------------------------------------------
# Training launcher
# --------------------------------------------------------------------------------------

def run_training(
    dataset_root: Path,
    dataset_yaml: Path,
    model_weights: str,
    *,
    epochs: int,
    batch: int,
    imgsz: int = 640,
    project: str = "runs/filtered_train",
    exp_name: str = "exp",
    device: str | int | None = None,
):
    print("🔥 Loading YOLO model …")
    model = YOLO(model_weights)
    print("✅ Model loaded – starting training")

    # Choose data argument
    data_arg = (
        str(dataset_root)
        if "cls" in Path(model_weights).stem or model.task == "classify"
        else str(dataset_yaml)
    )

    results = model.train(
        data=data_arg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=exp_name,
        device=device if device is not None else 0,
        pretrained=True,
    )

    print("\n🎉 Training finished")
    print(results)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def find_latest_best(project_dir: Path) -> Optional[Path]:
    """Return path to most recently modified *best.pt* under *project_dir* (or None)."""
    best_paths = []
    for exp_dir in project_dir.glob("**/weights/best.pt"):
        try:
            mtime = exp_dir.stat().st_mtime
            best_paths.append((mtime, exp_dir))
        except FileNotFoundError:
            continue
    if not best_paths:
        return None
    best_paths.sort(reverse=True)  # newest first
    return best_paths[0][1]


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on filtered dataset")
    parser.add_argument("--data_dir", type=str, default="./Kaggle/cupybara/yolo11_dataset_filtered", help="Root of filtered dataset")
    parser.add_argument("--model", type=str, default="best", help="Base model weights to fine-tune or 'auto_best' to pick latest best.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--project", type=str, default="runs/filtered_train", help="Ultralytics project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or cpu (default: auto)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    args = parser.parse_args()

    # Optionally replace model weight path with latest best.pt
    if args.model.lower() in {"auto_best", "best", "latest_best"}:
        latest_best = find_latest_best(Path(args.project))
        if latest_best is None:
            raise FileNotFoundError("No previous best.pt found under project directory. Run initial training first or pass --model path.")
        print(f"🔄 Using latest best model: {latest_best}")
        args.model = str(latest_best)

    dataset_root = Path(args.data_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset_yaml = create_dataset_yaml(dataset_root)

    run_training(
        dataset_root,
        dataset_yaml,
        args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        exp_name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main() 