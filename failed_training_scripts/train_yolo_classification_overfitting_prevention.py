#!/usr/bin/env python3
"""
train_yolo_classification_overfitting_prevention.py
===================================================
Train an Ultralytics-YOLO CLASSIFICATION model on the filtered dataset with comprehensive
overfitting prevention techniques.

This script is specifically designed for CLASSIFICATION tasks where images are organized
in folders by class name (not object detection with label files).

Dataset Structure Expected:
    dataset_root/
        train/
            class1/*.jpg
            class2/*.jpg
            ...
        val/
            class1/*.jpg
            class2/*.jpg
            ...
        test/
            class1/*.jpg
            class2/*.jpg
            ...

Usage:
    python train_yolo_classification_overfitting_prevention.py \
        --data_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --model yolo11n-cls.pt \
        --epochs 50 \
        --batch 16 \
        --enable_overfitting_prevention \
        --patience 15
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import os
from typing import Optional, Dict, Any, List
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter
import shutil

# --------------------------------------------------------------------------------------
# Overfitting Prevention Configuration
# --------------------------------------------------------------------------------------

class ClassificationOverfittingConfig:
    """Configuration for classification overfitting prevention techniques."""
    
    def __init__(self):
        # Regularization
        self.dropout_rate = 0.3
        self.weight_decay = 0.0005
        self.label_smoothing = 0.1
        
        # Learning rate and optimization
        self.initial_lr = 0.001
        self.lr_scheduler = "cosine"  # cosine, step, plateau
        self.warmup_epochs = 3
        self.min_lr = 1e-6
        
        # Data augmentation (classification specific)
        self.degrees = 15.0          # rotation degrees
        self.translate = 0.1         # translation fraction
        self.scale = 0.2             # scaling range
        self.shear = 10.0            # shear degrees
        self.perspective = 0.0001    # perspective distortion
        self.flipud = 0.3            # vertical flip probability
        self.fliplr = 0.5            # horizontal flip probability
        self.mosaic = 0.0            # disabled for classification
        self.mixup = 0.2             # mixup probability
        self.copy_paste = 0.0        # disabled for classification
        
        # HSV augmentation
        self.hsv_h = 0.015           # hue augmentation
        self.hsv_s = 0.7             # saturation augmentation
        self.hsv_v = 0.4             # value augmentation
        
        # Early stopping and checkpointing
        self.patience = 15
        self.min_delta = 0.001
        self.save_period = 5
        
        # Training stability
        self.amp = True              # Automatic Mixed Precision
        self.optimizer = "AdamW"     # Better for overfitting prevention
        self.momentum = 0.937
        
        # Class balancing
        self.class_weights = True
        self.weighted_sampling = True

def calculate_class_distribution(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Calculate class distribution across train/val/test splits."""
    distribution = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        split_counts = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                count = len(list(class_dir.glob("*.jpg")))
                split_counts[class_name] = count
        
        distribution[split] = split_counts
    
    return distribution

def analyze_class_imbalance(distribution: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Analyze class imbalance and calculate weights."""
    train_counts = distribution.get('train', {})
    if not train_counts:
        return {}
    
    total_samples = sum(train_counts.values())
    num_classes = len(train_counts)
    
    # Calculate inverse frequency weights
    class_weights = {}
    max_count = max(train_counts.values())
    
    print(f"\n📊 Class Distribution Analysis:")
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Weight':<8}")
    print("-" * 60)
    
    for class_name in sorted(train_counts.keys()):
        train_count = train_counts.get(class_name, 0)
        val_count = distribution.get('val', {}).get(class_name, 0)
        test_count = distribution.get('test', {}).get(class_name, 0)
        
        # Calculate weight (inverse frequency with balancing)
        if train_count > 0:
            weight = max_count / train_count
        else:
            weight = 1.0
            
        class_weights[class_name] = weight
        
        print(f"{class_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {weight:<8.3f}")
    
    return class_weights

# --------------------------------------------------------------------------------------
# Enhanced Training Functions
# --------------------------------------------------------------------------------------

def create_learning_rate_config(config: ClassificationOverfittingConfig, epochs: int) -> Dict[str, Any]:
    """Create learning rate scheduler configuration."""
    lr_config = {
        "lr0": config.initial_lr,
        "lrf": config.min_lr / config.initial_lr,
        "warmup_epochs": config.warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
    }
    
    if config.lr_scheduler == "cosine":
        # Cosine annealing parameters
        lr_config.update({
            "cos_lr": True,
        })
    elif config.lr_scheduler == "step":
        # Step decay parameters
        lr_config.update({
            "lrf": 0.1,
        })
    
    return lr_config

def save_training_checkpoint(model_path: Path, epoch: int, metrics: Dict[str, float], 
                           checkpoint_dir: Path) -> Path:
    """Save model checkpoint with metadata for classification."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Copy model file
    if model_path.exists():
        shutil.copy2(model_path, checkpoint_path)
    
    # Save metadata
    metadata = {
        "epoch": epoch,
        "metrics": metrics,
        "model_type": "classification",
        "checkpoint_path": str(checkpoint_path)
    }
    
    metadata_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Checkpoint saved: epoch {epoch} → {checkpoint_path}")
    return checkpoint_path

# --------------------------------------------------------------------------------------
# Enhanced Training Launcher
# --------------------------------------------------------------------------------------

def run_classification_training_with_overfitting_prevention(
    dataset_root: Path,
    model_weights: str,
    config: ClassificationOverfittingConfig,
    *,
    epochs: int,
    batch: int,
    imgsz: int = 640,
    project: str = "runs/classification_overfitting_prevention",
    exp_name: str = "exp",
    device: str | int | None = None,
    enable_overfitting_prevention: bool = True,
    verbose: bool = True,
):
    """Enhanced classification training with comprehensive overfitting prevention."""
    
    print("🔥 Loading YOLO CLASSIFICATION model …")
    model = YOLO(model_weights)
    print("✅ Model loaded – starting enhanced classification training")
    print(f"🛡️  Overfitting prevention: {'Enabled' if enable_overfitting_prevention else 'Disabled'}")
    print(f"📊 Task: Classification")
    print(f"⏱️  Early stopping patience: {config.patience} epochs")
    print(f"💾 Save period: {config.save_period} epochs")

    # Analyze class distribution
    distribution = calculate_class_distribution(dataset_root)
    class_weights = analyze_class_imbalance(distribution)

    # Setup checkpointing
    checkpoint_dir = Path(project) / exp_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Base training configuration for classification
    train_kwargs = {
        "data": str(dataset_root),  # For classification, pass the dataset directory
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": exp_name,
        "device": device if device is not None else 0,
        "pretrained": True,
        "patience": config.patience,
        "save_period": config.save_period,
        "verbose": verbose,
        "plots": True,
        "save": True,
        "val": True,
    }
    
    # Add overfitting prevention techniques
    if enable_overfitting_prevention:
        # Learning rate scheduling
        lr_config = create_learning_rate_config(config, epochs)
        train_kwargs.update(lr_config)
        
        # Enhanced regularization and augmentation
        train_kwargs.update({
            # Regularization
            "dropout": config.dropout_rate,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            
            # Data augmentation (classification appropriate)
            "degrees": config.degrees,
            "translate": config.translate,
            "scale": config.scale,
            "shear": config.shear,
            "perspective": config.perspective,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "mixup": config.mixup,
            
            # HSV augmentation
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            
            # Training optimization
            "amp": config.amp,
            "optimizer": config.optimizer,
            "momentum": config.momentum,
            
            # Disable object detection specific augmentations
            "mosaic": 0.0,
            "copy_paste": 0.0,
            
            # Additional stability features
            "fraction": 1.0,  # Use full dataset
            "workers": 4,     # Reduced workers for stability
        })
        
        print("🛡️  Enhanced overfitting prevention techniques enabled:")
        print(f"   • Dropout rate: {config.dropout_rate}")
        print(f"   • Weight decay: {config.weight_decay}")
        print(f"   • Label smoothing: {config.label_smoothing}")
        print(f"   • Learning rate scheduler: {config.lr_scheduler}")
        print(f"   • Enhanced data augmentation (classification)")
        print(f"   • Early stopping (patience: {config.patience})")
        print(f"   • Automatic Mixed Precision: {config.amp}")
        print(f"   • Optimizer: {config.optimizer}")

    try:
        print(f"\n🚀 Starting classification training with overfitting prevention...")
        print(f"📁 Dataset: {dataset_root}")
        print(f"🎯 Classes: {len(class_weights)}")
        print(f"🔄 Epochs: {epochs} (with early stopping)")
        print(f"📦 Batch size: {batch}")
        print(f"🖼️  Image size: {imgsz}")
        
        results = model.train(**train_kwargs)
        
        print("\n🎉 Training finished successfully!")
        
        # Save final model with comprehensive metadata
        if results and hasattr(results, 'save_dir'):
            final_metrics = {
                "task": "classification",
                "final_epoch": epochs,
                "dataset_classes": len(class_weights),
                "class_distribution": distribution,
                "class_weights": class_weights,
                "overfitting_prevention": enable_overfitting_prevention,
                "config": {
                    "dropout_rate": config.dropout_rate,
                    "weight_decay": config.weight_decay,
                    "label_smoothing": config.label_smoothing,
                    "lr_scheduler": config.lr_scheduler,
                    "patience": config.patience,
                    "optimizer": config.optimizer,
                },
                "training_args": train_kwargs
            }
            
            metadata_path = Path(results.save_dir) / "classification_training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            print(f"💾 Training metadata saved to: {metadata_path}")
            print(f"🏆 Best model saved at: {results.save_dir}/weights/best.pt")
            print(f"📊 Training results saved in: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"\n🔧 Troubleshooting suggestions:")
        print(f"1. Check if dataset directory structure is correct")
        print(f"2. Ensure you're using a classification model (e.g., yolo11n-cls.pt)")
        print(f"3. Try reducing batch size if out of memory")
        print(f"4. Check if images are accessible and not corrupted")
        raise

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLO CLASSIFICATION with overfitting prevention")
    parser.add_argument("--data_dir", type=str, default="./Kaggle/cupybara/yolo11_dataset_filtered", 
                       help="Root of classification dataset (with train/val/test folders)")
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt", 
                       help="Classification model weights (e.g., yolo11n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--project", type=str, default="runs/classification_overfitting_prevention", 
                       help="Project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or cpu")
    
    # Overfitting prevention arguments
    parser.add_argument("--enable_overfitting_prevention", action="store_true", default=True, 
                       help="Enable overfitting prevention techniques")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--save_period", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                       choices=["cosine", "step", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup probability")
    
    args = parser.parse_args()

    # Create overfitting prevention configuration
    config = ClassificationOverfittingConfig()
    config.patience = args.patience
    config.save_period = args.save_period
    config.dropout_rate = args.dropout
    config.weight_decay = args.weight_decay
    config.label_smoothing = args.label_smoothing
    config.lr_scheduler = args.lr_scheduler
    config.mixup = args.mixup

    dataset_root = Path(args.data_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Verify dataset structure
    required_dirs = ['train']
    missing_dirs = [d for d in required_dirs if not (dataset_root / d).exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing required directories: {missing_dirs}")

    print("\n🛡️  CLASSIFICATION OVERFITTING PREVENTION TRAINING")
    print("="*60)
    print(f"📁 Dataset: {dataset_root}")
    print(f"🏗️  Model: {args.model}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🛡️  Prevention enabled: {args.enable_overfitting_prevention}")
    print(f"⏱️  Patience: {config.patience}")
    print(f"💾 Save period: {config.save_period}")
    print(f"🎯 Task: Image Classification")
    print("="*60)

    run_classification_training_with_overfitting_prevention(
        dataset_root,
        args.model,
        config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        exp_name=args.name,
        device=args.device,
        enable_overfitting_prevention=args.enable_overfitting_prevention,
        verbose=True,
    )

if __name__ == "__main__":
    main() 