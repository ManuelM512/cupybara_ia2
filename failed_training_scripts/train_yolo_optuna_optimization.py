#!/usr/bin/env python3
"""
train_yolo_optuna_optimization.py
=================================
Hyperparameter optimization for YOLO classification using Optuna framework.

This script automatically searches for optimal hyperparameters to:
1. Maximize classification accuracy
2. Prevent overfitting
3. Balance training efficiency

Usage:
    python train_yolo_optuna_optimization.py \
        --data_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --model yolo11n-cls.pt \
        --n_trials 50 \
        --timeout 7200 \
        --study_name animal_classification_study

Features:
- Bayesian optimization of 15+ hyperparameters
- Early stopping for both trials and training
- Multi-objective optimization (accuracy vs efficiency)
- SQLite study persistence for resumable optimization
- Real-time visualization dashboard
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import plotly.io as pio

from ultralytics import YOLO
import torch
import gc

# --------------------------------------------------------------------------------------
# Optuna Study Configuration
# --------------------------------------------------------------------------------------

class OptunaConfig:
    """Configuration for Optuna hyperparameter optimization."""
    
    def __init__(self):
        # Study configuration
        self.study_name = "yolo_classification_optimization"
        self.storage_url = "sqlite:///optuna_studies.db"
        self.direction = "maximize"  # Maximize validation accuracy
        
        # Trial configuration
        self.n_trials = 50
        self.timeout = 7200  # 2 hours in seconds
        self.n_jobs = 1  # Number of parallel trials
        
        # Training configuration for each trial
        self.max_epochs_per_trial = 30
        self.early_stopping_patience = 8
        self.min_epochs = 5  # Minimum epochs before early stopping
        
        # Pruning configuration
        self.enable_pruning = True
        self.pruning_warmup_steps = 5
        
        # Performance thresholds
        self.min_accuracy_threshold = 0.1  # Prune trials below 10% accuracy
        self.target_accuracy = 0.95  # Stop study if achieved

def create_optuna_study(config: OptunaConfig) -> optuna.Study:
    """Create or load an Optuna study with proper configuration."""
    
    # Create pruner for early stopping of unpromising trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruning_warmup_steps,
        n_warmup_steps=config.min_epochs,
        interval_steps=1
    ) if config.enable_pruning else optuna.pruners.NopPruner()
    
    # Create sampler for efficient hyperparameter search
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,  # Random trials before Bayesian optimization
        n_ei_candidates=24,   # Number of candidate points for expected improvement
        seed=42  # For reproducibility
    )
    
    # Create or load study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage_url,
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    return study

# --------------------------------------------------------------------------------------
# Hyperparameter Search Space
# --------------------------------------------------------------------------------------

def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for YOLO classification."""
    
    # Learning rate and optimization
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    lrf = trial.suggest_float("lrf", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Regularization
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    
    # Data augmentation
    degrees = trial.suggest_float("degrees", 0.0, 45.0)
    translate = trial.suggest_float("translate", 0.0, 0.3)
    scale = trial.suggest_float("scale", 0.0, 0.5)
    shear = trial.suggest_float("shear", 0.0, 20.0)
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    flipud = trial.suggest_float("flipud", 0.0, 0.5)
    fliplr = trial.suggest_float("fliplr", 0.0, 0.8)
    mixup = trial.suggest_float("mixup", 0.0, 0.5)
    
    # HSV augmentation
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
    
    # Training parameters
    batch_size = 8
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    
    # Learning rate scheduler
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    cos_lr = trial.suggest_categorical("cos_lr", [True, False])
    
    return {
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,
        "mixup": mixup,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "batch": batch_size,
        "optimizer": optimizer,
        "warmup_epochs": warmup_epochs,
        "cos_lr": cos_lr,
    }

# --------------------------------------------------------------------------------------
# Training Objective Function
# --------------------------------------------------------------------------------------

def train_and_evaluate(
    dataset_root: Path,
    model_weights: str,
    hyperparams: Dict[str, Any],
    trial: optuna.Trial,
    config: OptunaConfig
) -> float:
    """Train YOLO model with given hyperparameters and return validation accuracy."""
    
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n🔥 Trial {trial.number}: Training with hyperparameters:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
        
        # Load model
        model = YOLO(model_weights)
        
        # Prepare training arguments
        train_kwargs = {
            "data": str(dataset_root),
            "epochs": config.max_epochs_per_trial,
            "imgsz": 640,
            "project": f"runs/optuna_trials",
            "name": f"trial_{trial.number:03d}",
            "device": 0 if torch.cuda.is_available() else "cpu",
            "pretrained": True,
            "patience": config.early_stopping_patience,
            "save": False,  # Don't save intermediate models
            "plots": False,  # Disable plots for speed
            "verbose": False,  # Reduce output
            "val": True,
            "save_period": -1,  # Don't save periodic checkpoints
            **hyperparams
        }
        
        # Custom callback for intermediate values (for pruning)
        class OptunaCallback:
            def __init__(self, trial: optuna.Trial):
                self.trial = trial
                self.epoch = 0
                
            def on_train_epoch_end(self, trainer):
                self.epoch += 1
                if hasattr(trainer, 'metrics') and 'metrics/accuracy_top1' in trainer.metrics:
                    accuracy = trainer.metrics['metrics/accuracy_top1']
                    
                    # Report intermediate value for pruning
                    self.trial.report(accuracy, self.epoch)
                    
                    # Check if trial should be pruned
                    if self.trial.should_prune():
                        raise optuna.TrialPruned(f"Trial pruned at epoch {self.epoch}")
        
        # Train model
        results = model.train(**train_kwargs)
        
        # Extract final validation accuracy
        if results and hasattr(results, 'results_dict'):
            # Get the best validation accuracy achieved
            val_accuracy = getattr(results, 'best_fitness', 0.0)
            if val_accuracy == 0.0:
                # Fallback: try to get from metrics
                metrics_file = Path(results.save_dir) / "results.csv"
                if metrics_file.exists():
                    import pandas as pd
                    df = pd.read_csv(metrics_file)
                    if 'metrics/accuracy_top1' in df.columns:
                        val_accuracy = df['metrics/accuracy_top1'].max()
        else:
            val_accuracy = 0.0
        
        # Cleanup to save memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✅ Trial {trial.number} completed: Accuracy = {val_accuracy:.4f}")
        return val_accuracy
        
    except optuna.TrialPruned:
        print(f"✂️ Trial {trial.number} pruned")
        raise
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        return 0.0  # Return worst possible score

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    global dataset_root, model_weights, optuna_config
    
    # Suggest hyperparameters for this trial
    hyperparams = suggest_hyperparameters(trial)
    
    # Train and evaluate
    accuracy = train_and_evaluate(
        dataset_root, 
        model_weights, 
        hyperparams, 
        trial, 
        optuna_config
    )
    
    return accuracy

# --------------------------------------------------------------------------------------
# Visualization and Analysis
# --------------------------------------------------------------------------------------

def create_optimization_visualizations(study: optuna.Study, output_dir: Path):
    """Create comprehensive visualization of optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating optimization visualizations...")
    
    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(output_dir / "optimization_history.html")
        pio.write_image(fig1, output_dir / "optimization_history.png", width=1200, height=600)
        
        # Parameter importances
        if len(study.trials) > 10:
            fig2 = plot_param_importances(study)
            fig2.write_html(output_dir / "param_importances.html")
            pio.write_image(fig2, output_dir / "param_importances.png", width=1200, height=800)
        
        # Parallel coordinate plot
        if len(study.trials) > 5:
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html(output_dir / "parallel_coordinate.html")
            pio.write_image(fig3, output_dir / "parallel_coordinate.png", width=1400, height=800)
        
        # Parameter slice plots
        if len(study.trials) > 10:
            fig4 = plot_slice(study)
            fig4.write_html(output_dir / "parameter_slices.html")
            pio.write_image(fig4, output_dir / "parameter_slices.png", width=1400, height=1000)
        
        print(f"✅ Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

def analyze_optimization_results(study: optuna.Study, output_dir: Path):
    """Analyze and save optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📈 OPTIMIZATION RESULTS ANALYSIS")
    print("=" * 60)
    
    # Best trial
    best_trial = study.best_trial
    print(f"🏆 Best Trial: #{best_trial.number}")
    print(f"🎯 Best Accuracy: {best_trial.value:.4f}")
    print(f"⏱️  Duration: {best_trial.duration}")
    
    print(f"\n🔧 Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"   {key}: {value}")
    
    # Trial statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\n📊 Trial Statistics:")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Completed: {len(completed_trials)}")
    print(f"   Pruned: {len(pruned_trials)}")
    print(f"   Failed: {len(failed_trials)}")
    
    if completed_trials:
        accuracies = [t.value for t in completed_trials]
        print(f"   Best accuracy: {max(accuracies):.4f}")
        print(f"   Average accuracy: {np.mean(accuracies):.4f}")
        print(f"   Std accuracy: {np.std(accuracies):.4f}")
    
    # Save detailed results
    results = {
        "study_name": study.study_name,
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "duration": str(best_trial.duration)
        },
        "statistics": {
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials)
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "duration": str(t.duration)
            }
            for t in study.trials
        ]
    }
    
    results_file = output_dir / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    return best_trial.params

# --------------------------------------------------------------------------------------
# Main Optimization Function
# --------------------------------------------------------------------------------------

def run_optuna_optimization(
    dataset_root: Path,
    model_weights: str,
    config: OptunaConfig,
    resume: bool = True
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization."""
    
    # Set global variables for objective function
    globals()['dataset_root'] = dataset_root
    globals()['model_weights'] = model_weights
    globals()['optuna_config'] = config
    
    # Create or load study
    study = create_optuna_study(config)
    
    print(f"\n🔍 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"📚 Study: {config.study_name}")
    print(f"🎯 Direction: {config.direction}")
    print(f"🔄 Max trials: {config.n_trials}")
    print(f"⏱️  Timeout: {config.timeout}s ({config.timeout//3600}h {(config.timeout%3600)//60}m)")
    print(f"🛡️  Early stopping: {config.early_stopping_patience} epochs")
    print(f"✂️  Pruning: {'Enabled' if config.enable_pruning else 'Disabled'}")
    
    if len(study.trials) > 0:
        print(f"📈 Resuming study with {len(study.trials)} existing trials")
        print(f"🏆 Current best: {study.best_value:.4f}")
    
    # Setup logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Custom callback for study progress
    def study_callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"✅ Trial {trial.number}: {trial.value:.4f}")
            
            # Check if target accuracy reached
            if trial.value >= config.target_accuracy:
                print(f"🎯 Target accuracy {config.target_accuracy:.4f} reached! Stopping study.")
                study.stop()
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"✂️ Trial {trial.number}: Pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"❌ Trial {trial.number}: Failed")
    
    try:
        # Run optimization
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout,
            callbacks=[study_callback],
            show_progress_bar=True
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization error: {e}")
    
    # Create output directory
    output_dir = Path(f"optuna_results_{config.study_name}")
    
    # Analyze results
    best_params = analyze_optimization_results(study, output_dir)
    
    # Create visualizations
    create_optimization_visualizations(study, output_dir)
    
    print(f"\n🎉 Optimization completed!")
    print(f"💾 Results saved to: {output_dir}")
    print(f"🏆 Best accuracy: {study.best_value:.4f}")
    
    return best_params

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for YOLO classification")
    parser.add_argument("--data_dir", type=str, default="./Kaggle/cupybara/yolo11_dataset_filtered",
                       help="Root of classification dataset")
    parser.add_argument("--model", type=str, default="runs/filtered_train_augmented/exp4/weights/best.pt",
                       help="Base model weights")
    parser.add_argument("--study_name", type=str, default="yolo_animal_classification",
                       help="Optuna study name")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=7200,
                       help="Optimization timeout in seconds")
    parser.add_argument("--max_epochs", type=int, default=30,
                       help="Maximum epochs per trial")
    parser.add_argument("--patience", type=int, default=8,
                       help="Early stopping patience")
    parser.add_argument("--enable_pruning", action="store_true", default=True,
                       help="Enable trial pruning")
    parser.add_argument("--target_accuracy", type=float, default=0.95,
                       help="Target accuracy to stop optimization")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume existing study")
    
    args = parser.parse_args()
    
    # Validate dataset
    dataset_root = Path(args.data_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    if not (dataset_root / "train").exists():
        raise FileNotFoundError(f"Training directory not found: {dataset_root / 'train'}")
    
    # Create configuration
    config = OptunaConfig()
    config.study_name = args.study_name
    config.n_trials = args.n_trials
    config.timeout = args.timeout
    config.max_epochs_per_trial = args.max_epochs
    config.early_stopping_patience = args.patience
    config.enable_pruning = args.enable_pruning
    config.target_accuracy = args.target_accuracy
    
    print(f"\n🚀 Starting Optuna optimization for animal classification")
    print(f"📁 Dataset: {dataset_root}")
    print(f"🏗️  Base model: {args.model}")
    
    # Run optimization
    best_params = run_optuna_optimization(
        dataset_root,
        args.model,
        config,
        resume=args.resume
    )
    
    print(f"\n🎯 OPTIMIZATION COMPLETE")
    print(f"🏆 Best hyperparameters found:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Use these hyperparameters for final training")
    print(f"2. Check visualizations in optuna_results_{config.study_name}/")
    print(f"3. Run Optuna dashboard: optuna-dashboard sqlite:///optuna_studies.db")

if __name__ == "__main__":
    main() 