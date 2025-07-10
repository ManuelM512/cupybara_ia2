#!/usr/bin/env python3
"""
train_yolo_filtered_augmented.py
--------------------------------
Train an Ultralytics-YOLO detection model on the **filtered** dataset with comprehensive
data augmentation, including infrared filter simulation.

This script extends the base training functionality with:
1. Advanced data augmentation pipeline
2. Infrared filter simulation
3. Configurable augmentation parameters
4. Augmentation visualization (optional)

Assumptions
===========
1. Dataset folder follows standard YOLO directory layout::

       <dataset_root>/
           dataset.yaml            # already present (if missing we create it)
           train/<species>/*.jpg
           val/<species>/*.jpg
           test/<species>/*.jpg

2. Ultralytics-YOLO (>=8.1) is installed (already in requirements.txt).

Quick start (train with augmentation)::

    python train_yolo_filtered_augmented.py \
            --data_dir Kaggle/cupybara/yolo11_dataset_filtered \
            --model yolov8n.pt \
            --epochs 100 \
            --batch 32 \
            --imgsz 640 \
            --enable_augmentation \
            --enable_infrared \
            --patience 10

The script will:
1. Auto-generate `dataset.yaml` if not found (based on sub-folders under *train/*).
2. Apply data augmentation including infrared filter if enabled.
3. Launch training with `ultralytics.YOLO`.
4. Save runs to *runs/filtered_train_augmented/* (can be customised).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import os
from typing import Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import torch
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Data Augmentation Utilities
# --------------------------------------------------------------------------------------

def create_infrared_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply infrared filter simulation to an image.
    
    Infrared filter typically:
    1. Enhances red channel
    2. Suppresses blue and green channels
    3. Increases contrast
    4. May add some noise/grain
    """
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Enhance red channel (infrared is sensitive to red)
    img_float[:, :, 2] *= 1.5  # Red channel
    img_float[:, :, 1] *= 0.7  # Green channel (suppress)
    img_float[:, :, 0] *= 0.6  # Blue channel (suppress)
    
    # Increase contrast
    img_float = np.clip(img_float, 0, 1)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    
    # Add slight noise for realistic effect
    noise = np.random.normal(0, 0.02, img_float.shape)
    img_float = np.clip(img_float + noise, 0, 1)
    
    # Convert back to uint8
    return (img_float * 255).astype(np.uint8)

def create_augmentation_pipeline(
    enable_infrared: bool = False,
    infrared_prob: float = 0.3,
    **kwargs
) -> A.Compose:
    """
    Create a comprehensive data augmentation pipeline.
    
    Args:
        enable_infrared: Whether to include infrared filter
        infrared_prob: Probability of applying infrared filter
        **kwargs: Other augmentation parameters
    """
    transforms = []
    
    # Basic geometric transformations
    transforms.extend([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.Affine(rotate=10, p=0.5),
            A.Affine(rotate=-10, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.3),
    ])
    
    # Color and brightness augmentations
    transforms.extend([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.3),
    ])
    
    # Noise and blur effects
    transforms.extend([
        A.GaussNoise(std_range=(0.2, 0.4), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.3),
    ])
    
    # Weather and environmental effects (simplified)
    transforms.extend([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        ], p=0.2),
    ])
    
    # Infrared filter (custom transformation)
    if enable_infrared:
        transforms.append(
            A.Lambda(
                name="infrared_filter",
                image=create_infrared_filter,
                p=infrared_prob
            )
        )
    
    # Final transformations
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def visualize_augmentations(
    image_path: Path,
    augmentation_pipeline: A.Compose,
    num_samples: int = 5,
    output_dir: Optional[Path] = None
):
    """
    Visualize augmentation effects on a sample image.
    
    Args:
        image_path: Path to sample image
        augmentation_pipeline: Augmentation pipeline to test
        num_samples: Number of augmented samples to generate
        output_dir: Directory to save visualization (optional)
    """
    if not image_path.exists():
        print(f"⚠️  Sample image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"🔍 Visualizing augmentations on {image_path.name}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    for i in range(num_samples):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        # Apply augmentation
        augmented = augmentation_pipeline(image=image)['image']
        
        # Convert back to displayable format
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.permute(1, 2, 0).numpy()
            augmented = (augmented * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        
        axes[row, col].imshow(augmented)
        axes[row, col].set_title(f"Augmented {i+1}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"augmentation_samples_{image_path.stem}.png", dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to {output_dir}")
    
    plt.show()

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
    project: str = "runs/filtered_train_augmented",
    exp_name: str = "exp",
    device: str | int | None = None,
    enable_augmentation: bool = False,
    enable_infrared: bool = False,
    infrared_prob: float = 0.3,
    visualize_aug: bool = False,
    patience: int = 10,
):
    print("🔥 Loading YOLO model …")
    model = YOLO(model_weights)
    print("✅ Model loaded – starting training")
    print(f"⏱️  Early stopping patience: {patience} epochs")

    # Create augmentation pipeline if enabled
    if enable_augmentation:
        print("🎨 Creating data augmentation pipeline...")
        augmentation_pipeline = create_augmentation_pipeline(
            enable_infrared=enable_infrared,
            infrared_prob=infrared_prob
        )
        print(f"✅ Augmentation pipeline created (infrared: {enable_infrared})")
        
        # Visualize augmentations if requested
        if visualize_aug:
            try:
                import matplotlib.pyplot as plt
                # Find a sample image
                train_dir = dataset_root / "train"
                sample_class = next(train_dir.iterdir())
                sample_image = next(sample_class.glob("*.jpg"))
                visualize_augmentations(
                    sample_image,
                    augmentation_pipeline,
                    output_dir=Path(project) / exp_name
                )
            except ImportError:
                print("⚠️  matplotlib not available, skipping visualization")
            except Exception as e:
                print(f"⚠️  Could not visualize augmentations: {e}")

    # Choose data argument
    data_arg = (
        str(dataset_root)
        if "cls" in Path(model_weights).stem or model.task == "classify"
        else str(dataset_yaml)
    )

    # Training configuration
    train_kwargs = {
        "data": data_arg,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": exp_name,
        "device": device if device is not None else 0,
        "pretrained": True,
        "patience": patience,  # Early stopping patience
    }
    
    # Add augmentation if enabled
    if enable_augmentation:
        train_kwargs.update({
            "augment": True,  # Enable YOLO's built-in augmentation
            "mosaic": 0.5,    # Mosaic augmentation
            "mixup": 0.1,     # Mixup augmentation
            "copy_paste": 0.1, # Copy-paste augmentation
        })
        print("🎯 Training with data augmentation enabled")

    results = model.train(**train_kwargs)

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
    parser = argparse.ArgumentParser(description="Train YOLO on filtered dataset with data augmentation")
    parser.add_argument("--data_dir", type=str, default="./Kaggle/cupybara/yolo11_dataset_filtered", help="Root of filtered dataset")
    parser.add_argument("--model", type=str, default="best", help="Base model weights to fine-tune or 'auto_best' to pick latest best.pt")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--project", type=str, default="runs/filtered_train_augmented", help="Ultralytics project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or cpu (default: auto)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    

    # Augmentation arguments
    parser.add_argument("--enable_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--enable_infrared", action="store_true", help="Enable infrared filter in augmentation")
    parser.add_argument("--infrared_prob", type=float, default=0.3, help="Probability of applying infrared filter")
    parser.add_argument("--visualize_aug", action="store_true", help="Visualize augmentation effects before training")
    
    args = parser.parse_args()

    # Optionally replace model weight path with latest best.pt
    if args.model.lower() in {"auto_best", "best", "latest_best"}:
        latest_best = find_latest_best(Path("runs/filtered_train_augmented/exp3"))
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
        enable_augmentation=args.enable_augmentation,
        enable_infrared=args.enable_infrared,
        infrared_prob=args.infrared_prob,
        visualize_aug=args.visualize_aug,
        patience=args.patience,
    )


if __name__ == "__main__":
    main() 