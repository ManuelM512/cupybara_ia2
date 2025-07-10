#!/usr/bin/env python3
"""
train_with_optimized_params.py
==============================
Train the final YOLO classification model using optimal hyperparameters found by Optuna.

This script loads the best hyperparameters from Optuna optimization and runs
a full training session with proper validation and testing.

Usage:
    python train_with_optimized_params.py \
        --optuna_results optuna_results_yolo_animal_classification/optimization_results.json \
        --data_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --model yolo11n-cls.pt \
        --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import shutil

from ultralytics import YOLO
import torch

def load_best_hyperparameters(results_file: Path) -> Dict[str, Any]:
    """Load the best hyperparameters from Optuna optimization results."""
    
    if not results_file.exists():
        raise FileNotFoundError(f"Optuna results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_trial']['params']
    best_accuracy = results['best_trial']['value']
    
    print(f"📊 Loading optimized hyperparameters:")
    print(f"🏆 Best validation accuracy: {best_accuracy:.4f}")
    print(f"🔧 Best parameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    return best_params

def run_final_training(
    dataset_root: Path,
    model_weights: str,
    best_params: Dict[str, Any],
    epochs: int = 100,
    project: str = "runs/final_optimized_training",
    exp_name: str = "optimized_model"
) -> str:
    """Run final training with optimized hyperparameters."""
    
    print(f"\n🚀 Starting final training with optimized hyperparameters...")
    print(f"📁 Dataset: {dataset_root}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {best_params.get('batch', 16)}")
    
    # Load model
    model = YOLO(model_weights)
    
    # Prepare training arguments with optimized parameters
    train_kwargs = {
        "data": str(dataset_root),
        "epochs": epochs,
        "imgsz": 640,
        "project": project,
        "name": exp_name,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "pretrained": True,
        "patience": 20,  # Longer patience for final training
        "save": True,
        "plots": True,
        "verbose": True,
        "val": True,
        "save_period": 10,  # Save checkpoints every 10 epochs
        **best_params  # Use all optimized hyperparameters
    }
    
    print(f"\n🎯 Training configuration:")
    for key, value in train_kwargs.items():
        print(f"   {key}: {value}")
    
    # Train model
    results = model.train(**train_kwargs)
    
    print(f"\n🎉 Final training completed!")
    
    if results:
        save_dir = results.save_dir
        best_model_path = Path(save_dir) / "weights" / "best.pt"
        last_model_path = Path(save_dir) / "weights" / "last.pt"
        
        print(f"💾 Models saved:")
        print(f"   Best model: {best_model_path}")
        print(f"   Last model: {last_model_path}")
        
        # Save training metadata
        metadata = {
            "optimization_source": "optuna",
            "best_hyperparameters": best_params,
            "training_config": train_kwargs,
            "model_paths": {
                "best": str(best_model_path),
                "last": str(last_model_path)
            }
        }
        
        metadata_file = Path(save_dir) / "optimized_training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📋 Training metadata saved: {metadata_file}")
        
        return str(best_model_path)
    
    return ""

def evaluate_final_model(model_path: str, dataset_root: Path):
    """Evaluate the final trained model on test set."""
    
    if not Path(model_path).exists():
        print(f"⚠️  Model not found: {model_path}")
        return
    
    print(f"\n🧪 Evaluating final model on test set...")
    
    model = YOLO(model_path)
    
    # Run validation on test set
    test_results = model.val(
        data=str(dataset_root),
        split="test",
        save=True,
        plots=True,
        verbose=True
    )
    
    if test_results:
        print(f"🎯 Test Results:")
        if hasattr(test_results, 'top1'):
            print(f"   Top-1 Accuracy: {test_results.top1:.4f}")
        if hasattr(test_results, 'top5'):
            print(f"   Top-5 Accuracy: {test_results.top5:.4f}")
        
        print(f"📊 Detailed results saved in: {test_results.save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train final model with Optuna-optimized hyperparameters")
    parser.add_argument("--optuna_results", type=str, required=True,
                       help="Path to Optuna optimization results JSON file")
    parser.add_argument("--data_dir", type=str, default="./Kaggle/cupybara/yolo11_dataset_filtered",
                       help="Root of classification dataset")
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt",
                       help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs for final model")
    parser.add_argument("--project", type=str, default="runs/final_optimized_training",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="optimized_model",
                       help="Experiment name")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate final model on test set")
    
    args = parser.parse_args()
    
    # Validate inputs
    results_file = Path(args.optuna_results)
    dataset_root = Path(args.data_dir)
    
    if not results_file.exists():
        raise FileNotFoundError(f"Optuna results file not found: {results_file}")
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    print(f"\n🛠️  FINAL MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)
    print(f"📊 Using optimization results from: {results_file}")
    print(f"📁 Dataset: {dataset_root}")
    print(f"🏗️  Base model: {args.model}")
    print(f"🔄 Epochs: {args.epochs}")
    print("=" * 70)
    
    # Load best hyperparameters
    best_params = load_best_hyperparameters(results_file)
    
    # Run final training
    final_model_path = run_final_training(
        dataset_root,
        args.model,
        best_params,
        epochs=args.epochs,
        project=args.project,
        exp_name=args.name
    )
    
    # Evaluate final model
    if args.evaluate and final_model_path:
        evaluate_final_model(final_model_path, dataset_root)
    
    print(f"\n✅ FINAL TRAINING COMPLETE")
    print(f"🏆 Optimized model trained successfully!")
    if final_model_path:
        print(f"💾 Best model saved: {final_model_path}")
    print(f"📊 Use this model for inference and deployment")

if __name__ == "__main__":
    main() 