#!/usr/bin/env python3
"""
train_yolo11_crops_to_frames.py
===============================
Train a YOLOv11 model on extracted crops and validate against whole frames.

This script implements a sophisticated training strategy:
1. **Training**: Uses extracted crops for pure classification learning
2. **Validation**: Tests on whole frames from yolo11_dataset_filtered (detection + classification)

This approach allows the model to:
- Learn clean classification on isolated animal crops
- Be evaluated on real-world detection scenarios with full frames

Requirements:
- Extracted crops directory (from extract_crops_from_predictions.py)
- Original frame dataset (yolo11_dataset_filtered)
- YOLOv11 model weights

Usage:
    python train_yolo11_crops_to_frames.py \
        --crops_dir crops_from_predictions \
        --frames_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --model yolo11n-cls.pt \
        --epochs 100 \
        --batch 32 \
        --enable_augmentation
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_crops_to_frames.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CropsToFramesTrainer:
    """Train on crops, validate on frames trainer."""
    
    def __init__(self, crops_dir: Path, frames_dir: Path, output_dir: Path):
        """
        Initialize the trainer.
        
        Args:
            crops_dir: Directory with extracted crops organized by species
            frames_dir: Directory with original frames (yolo11_dataset_filtered structure)
            output_dir: Output directory for training artifacts
        """
        self.crops_dir = Path(crops_dir)
        self.frames_dir = Path(frames_dir) 
        self.output_dir = Path(output_dir)
        self.temp_dir = None
        
        # Validate directories
        if not self.crops_dir.exists():
            raise FileNotFoundError(f"Crops directory not found: {self.crops_dir}")
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("CropsToFramesTrainer initialized")
        logger.info(f"Crops dir: {self.crops_dir}")
        logger.info(f"Frames dir: {self.frames_dir}")
        logger.info(f"Output dir: {self.output_dir}")
    
    def discover_classes(self) -> List[str]:
        """Discover class names from crops directory."""
        crops_classes = []
        for class_dir in self.crops_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                crops_classes.append(class_dir.name)
        
        # Also check frames directory for consistency
        frames_train_dir = self.frames_dir / "train"
        frames_classes = []
        if frames_train_dir.exists():
            for class_dir in frames_train_dir.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    frames_classes.append(class_dir.name)
        
        # Use intersection to ensure consistency
        if frames_classes:
            common_classes = sorted(set(crops_classes) & set(frames_classes))
            if not common_classes:
                logger.warning("No common classes found between crops and frames!")
                common_classes = sorted(crops_classes)
        else:
            common_classes = sorted(crops_classes)
        
        logger.info(f"Discovered {len(common_classes)} classes: {common_classes}")
        return common_classes
    
    def create_crops_dataset_structure(self, classes: List[str], train_split: float = 0.8) -> Path:
        """
        Create YOLO-compatible dataset structure from crops.
        
        Args:
            classes: List of class names
            train_split: Fraction of crops to use for training
            
        Returns:
            Path to created dataset directory
        """
        # Create temporary dataset directory
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="crops_dataset_"))
        
        dataset_dir = self.temp_dir / "crops_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        logger.info("Creating crops dataset structure...")
        
        # Process each class
        total_crops = 0
        train_crops = 0
        val_crops = 0
        
        for class_name in tqdm(classes, desc="Processing classes"):
            crop_class_dir = self.crops_dir / class_name
            if not crop_class_dir.exists():
                logger.warning(f"Class directory not found in crops: {class_name}")
                continue
            
            # Get all crop images
            crop_files = list(crop_class_dir.glob("*.jpg")) + list(crop_class_dir.glob("*.png"))
            if not crop_files:
                logger.warning(f"No crop images found for class: {class_name}")
                continue
            
            # Shuffle and split
            import random
            random.shuffle(crop_files)
            split_idx = int(len(crop_files) * train_split)
            
            train_files = crop_files[:split_idx]
            val_files = crop_files[split_idx:]
            
            # Create class directories
            train_class_dir = train_dir / class_name
            val_class_dir = val_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Copy/link files
            for crop_file in train_files:
                dest = train_class_dir / crop_file.name
                shutil.copy2(crop_file, dest)
                train_crops += 1
            
            for crop_file in val_files:
                dest = val_class_dir / crop_file.name
                shutil.copy2(crop_file, dest)
                val_crops += 1
            
            total_crops += len(crop_files)
            logger.debug(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val")
        
        logger.info(f"Created crops dataset: {total_crops} total ({train_crops} train, {val_crops} val)")
        
        # Create dataset.yaml
        dataset_yaml = dataset_dir / "dataset.yaml"
        yaml_config = {
            "path": str(dataset_dir.resolve()),
            "train": "train",
            "val": "val",
            "names": classes,
            "nc": len(classes)
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_config, f)
        
        logger.info(f"Created dataset.yaml: {dataset_yaml}")
        return dataset_dir
    
    def create_augmentation_pipeline(self, enable_infrared: bool = False) -> A.Compose:
        """Create augmentation pipeline for crops training."""
        transforms = [
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            
            # Color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            
            # Advanced augmentations
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ]
        
        # Add infrared simulation if enabled
        if enable_infrared:
            def infrared_transform(image, **kwargs):
                """Simple infrared filter simulation."""
                img_float = image.astype(np.float32) / 255.0
                # Enhance red, suppress blue/green
                img_float[:, :, 2] *= 1.4  # Red
                img_float[:, :, 1] *= 0.8  # Green
                img_float[:, :, 0] *= 0.7  # Blue
                img_float = np.clip(img_float, 0, 1)
                return (img_float * 255).astype(np.uint8)
            
            transforms.append(A.Lambda(image=infrared_transform, p=0.3))
        
        return A.Compose(transforms)
    
    def train_on_crops(
        self,
        model_path: str,
        classes: List[str],
        epochs: int = 100,
        batch_size: int = 32,
        imgsz: int = 224,
        enable_augmentation: bool = True,
        enable_infrared: bool = False,
        patience: int = 10,
        **kwargs
    ) -> YOLO:
        """
        Train YOLO classification model on crops.
        
        Args:
            model_path: Path to base model weights
            classes: List of class names
            epochs: Number of training epochs
            batch_size: Training batch size
            imgsz: Image size for training
            enable_augmentation: Whether to enable data augmentation
            enable_infrared: Whether to enable infrared simulation
            patience: Early stopping patience
            
        Returns:
            Trained YOLO model
        """
        logger.info("Starting training on crops...")
        
        # Create crops dataset structure
        crops_dataset_dir = self.create_crops_dataset_structure(classes)
        
        # Initialize model
        model = YOLO(model_path)
        logger.info(f"Loaded model: {model_path}")
        
        # Training configuration
        train_kwargs = {
            "data": str(crops_dataset_dir),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "project": str(self.output_dir),
            "name": "crops_training",
            "patience": patience,
            "device": 0 if torch.cuda.is_available() else "cpu",
            "pretrained": True,
            "save": True,
            "save_period": 10,
        }
        
        # Add augmentation settings
        if enable_augmentation:
            train_kwargs.update({
                "augment": True,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 15.0,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.5,
                "flipud": 0.1,
                "mosaic": 0.0,  # Disable mosaic for classification
                "mixup": 0.0,   # Disable mixup for classification
            })
            logger.info("Data augmentation enabled")
        
        # Start training
        logger.info("Starting training...")
        results = model.train(**train_kwargs)
        
        logger.info("Crops training completed!")
        return model
    
    def validate_on_frames(
        self,
        model: YOLO,
        classes: List[str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_images_per_split: Optional[int] = None
    ) -> Dict:
        """
        Validate trained model on whole frames using detection.
        
        Args:
            model: Trained YOLO model
            classes: List of class names
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            max_images_per_split: Maximum images to test per split (for speed)
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting validation on whole frames...")
        
        results = {
            "train": {"total": 0, "correct": 0, "class_stats": {}},
            "val": {"total": 0, "correct": 0, "class_stats": {}},
            "test": {"total": 0, "correct": 0, "class_stats": {}}
        }
        
        # Initialize class stats
        for split in results:
            results[split]["class_stats"] = {cls: {"total": 0, "correct": 0} for cls in classes}
        
        # Process each split
        for split in ["train", "val", "test"]:
            split_dir = self.frames_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            logger.info(f"Processing {split} split...")
            
            # Collect all images by class
            split_images = []
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name not in classes:
                    continue
                
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                for img_path in images:
                    split_images.append((img_path, class_dir.name))
            
            # Limit images if specified
            if max_images_per_split and len(split_images) > max_images_per_split:
                import random
                split_images = random.sample(split_images, max_images_per_split)
                logger.info(f"Limited to {max_images_per_split} images for {split}")
            
            logger.info(f"Found {len(split_images)} images in {split}")
            
            # Process images
            for img_path, true_class in tqdm(split_images, desc=f"Validating {split}"):
                try:
                    # Run detection
                    detection_results = model(str(img_path), 
                                           conf=confidence_threshold,
                                           iou=iou_threshold,
                                           verbose=False)
                    
                    # Check if any detections match the true class
                    predicted_correctly = False
                    
                    for result in detection_results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            # Get class predictions
                            class_ids = result.boxes.cls.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            
                            for class_id, conf in zip(class_ids, confidences):
                                predicted_class = classes[int(class_id)]
                                if predicted_class == true_class:
                                    predicted_correctly = True
                                    break
                    
                    # Update statistics
                    results[split]["total"] += 1
                    results[split]["class_stats"][true_class]["total"] += 1
                    
                    if predicted_correctly:
                        results[split]["correct"] += 1
                        results[split]["class_stats"][true_class]["correct"] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
        
        # Calculate accuracies
        for split in results:
            if results[split]["total"] > 0:
                results[split]["accuracy"] = results[split]["correct"] / results[split]["total"]
            else:
                results[split]["accuracy"] = 0.0
            
            # Calculate per-class accuracies
            for cls in classes:
                class_total = results[split]["class_stats"][cls]["total"]
                if class_total > 0:
                    class_correct = results[split]["class_stats"][cls]["correct"]
                    results[split]["class_stats"][cls]["accuracy"] = class_correct / class_total
                else:
                    results[split]["class_stats"][cls]["accuracy"] = 0.0
        
        logger.info("Frame validation completed!")
        return results
    
    def save_results(self, results: Dict, model_path: Path):
        """Save validation results to files."""
        results_file = self.output_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary_file = self.output_dir / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Crops-to-Frames Training & Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Training: Crops from {self.crops_dir}\n")
            f.write(f"Validation: Frames from {self.frames_dir}\n\n")
            
            for split in ["train", "val", "test"]:
                if results[split]["total"] > 0:
                    f.write(f"{split.upper()} Results:\n")
                    f.write(f"  Overall Accuracy: {results[split]['accuracy']:.3f} ")
                    f.write(f"({results[split]['correct']}/{results[split]['total']})\n")
                    
                    f.write("  Per-class Accuracy:\n")
                    for cls, stats in results[split]["class_stats"].items():
                        if stats["total"] > 0:
                            f.write(f"    {cls}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})\n")
                    f.write("\n")
        
        logger.info(f"Results saved to {results_file} and {summary_file}")
    
    def cleanup(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directories")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on crops, validate on frames")
    
    # Data arguments
    parser.add_argument("--crops_dir", type=str, default="crops_from_predictions",
                       help="Directory with extracted crops organized by species")
    parser.add_argument("--frames_dir", type=str, default="Kaggle/cupybara/yolo11_dataset_filtered",
                       help="Directory with original frames (train/val/test structure)")
    parser.add_argument("--output_dir", type=str, default="runs/crops_to_frames",
                       help="Output directory for training artifacts")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt",
                       help="Base model weights (classification model)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of crops for training")
    
    # Augmentation arguments
    parser.add_argument("--enable_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--enable_infrared", action="store_true", help="Enable infrared simulation")
    
    # Validation arguments
    parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max_val_images", type=int, default=None, help="Max images per split for validation (for speed)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CropsToFramesTrainer(
        crops_dir=args.crops_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )
    
    try:
        # Discover classes
        classes = trainer.discover_classes()
        if not classes:
            raise ValueError("No classes found in crops directory")
        
        # Train on crops
        model = trainer.train_on_crops(
            model_path=args.model,
            classes=classes,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            enable_augmentation=args.enable_augmentation,
            enable_infrared=args.enable_infrared,
            patience=args.patience
        )
        
        # Load best model for validation
        best_model_path = Path(args.output_dir) / "crops_training" / "weights" / "best.pt"
        if best_model_path.exists():
            model = YOLO(str(best_model_path))
            logger.info(f"Loaded best model for validation: {best_model_path}")
        
        # Validate on frames
        results = trainer.validate_on_frames(
            model=model,
            classes=classes,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            max_images_per_split=args.max_val_images
        )
        
        # Save results
        trainer.save_results(results, best_model_path)
        
        # Print summary
        print("\nTraining and Validation Complete!")
        print(f"Results Summary:")
        for split in ["train", "val", "test"]:
            if results[split]["total"] > 0:
                acc = results[split]["accuracy"]
                total = results[split]["total"]
                correct = results[split]["correct"]
                print(f"   {split.upper()}: {acc:.3f} accuracy ({correct}/{total})")
        
        print(f"\nAll artifacts saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 