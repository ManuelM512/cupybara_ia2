#!/usr/bin/env python3
"""
megadetector_bird_frame_extractor.py
====================================
Extract frames containing animals from videos labeled as 'bird' in test.csv
--------------------------------------------------------------------

This script processes only videos from `dataset/test` that are labeled as "bird" 
in `test.csv`. It uses MegaDetector v6 to detect animals in each frame and saves
frames that contain animal detections to the existing YOLO dataset structure
with 80/10/10 video-level splitting.

Frames are saved to:
    <output_dir>/
        train/bird/*.jpg   # 80% of bird videos
        val/bird/*.jpg     # 10% of bird videos  
        test/bird/*.jpg    # 10% of bird videos

All frames from the same video stay in the same split to prevent data leakage.

Usage:
------
```bash
python megadetector_bird_frame_extractor.py \
        --csv_path test.csv \
        --videos_dir dataset/test \
        --output_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --confidence 0.70 \
        --frames_per_video 130 \
        --frame_interval 1
```

Dependencies (already in requirements.txt):
    • opencv-python, numpy, torch, ultralytics, tqdm
    • pytorch-wildlife (provides MegaDetectorV6)
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import torch
from PytorchWildlife.models import detection as pw_detection

# --------------------------------------------------------------------------------------
# CSV and splitting utilities
# --------------------------------------------------------------------------------------

def read_csv_for_species(csv_path: Path, target_species: str) -> List[str]:
    """Return list of video filenames that match the target species."""
    matching_videos = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            filename = row[0].strip().strip("'\"")
            species = row[1].strip().strip("'\"") if len(row) > 1 else "unknown"
            if species.lower() == target_species.lower():
                matching_videos.append(filename)
    return matching_videos


def split_videos_80_10_10(videos: List[str], rng: random.Random) -> Dict[str, List[str]]:
    """Split videos into train/val/test with 80/10/10 distribution."""
    rng.shuffle(videos)
    n = len(videos)
    
    if n == 0:
        return {"train": [], "val": [], "test": []}
    
    if n < 3:
        # For 1-2 videos, put in train and val
        return {
            "train": videos[:max(1, n-1)],
            "val": videos[max(1, n-1):],
            "test": []
        }
    
    # For 3-9 videos: guarantee at least 1 in val, rest split proportionally
    if n <= 9:
        val_size = 1
        test_size = max(0, n - 6)  # 0 for n<=6, 1 for n=7-8, 2 for n=9
        train_size = n - val_size - test_size
    else:
        # Standard percentage split for larger datasets
        train_size = int(round(n * 0.8))
        val_size = int(round(n * 0.1))
        test_size = n - train_size - val_size
    
    train_cut = train_size
    val_cut = train_cut + val_size
    
    return {
        "train": videos[:train_cut],
        "val": videos[train_cut:val_cut], 
        "test": videos[val_cut:]
    }

# --------------------------------------------------------------------------------------
# Frame processing utilities
# --------------------------------------------------------------------------------------

def get_frame_indices(total_frames: int, frames_per_video: int, frame_interval: int) -> List[int]:
    """Select frame indices to sample from video."""
    if total_frames <= 0:
        return []
    
    if total_frames <= frames_per_video:
        # Sample every frame_interval frames
        return list(range(0, total_frames, max(1, frame_interval)))
    else:
        # Evenly distribute frames_per_video samples
        return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()


def save_frame_to_disk(frame: np.ndarray, output_dir: Path, video_stem: str, frame_idx: int):
    """Save frame as JPEG to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(output_dir / filename), frame)

# --------------------------------------------------------------------------------------
# Main processing function
# --------------------------------------------------------------------------------------

def process_bird_videos(
    video_splits: Dict[str, List[str]],
    videos_dir: Path,
    output_dir: Path,
    model: pw_detection.MegaDetectorV6,
    *,
    confidence_thresh: float,
    frames_per_video: int,
    frame_interval: int
):
    """Process bird videos and extract frames with animal detections."""
    total_frames_saved = 0
    
    for split, video_list in video_splits.items():
        if not video_list:
            continue
            
        split_output_dir = output_dir / split / "bird"
        
        for video_name in tqdm(video_list, desc=f"Processing {split} videos", leave=False):
            # Try to find video file
            video_path = videos_dir / f"{video_name}.mp4"
            if not video_path.exists():
                # Try other extensions
                alternatives = list(videos_dir.glob(f"{video_name}.*"))
                if alternatives:
                    video_path = alternatives[0]
                else:
                    tqdm.write(f"⚠️  Video not found: {video_name}")
                    continue
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                tqdm.write(f"⚠️  Cannot open video: {video_path.name}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = get_frame_indices(total_frames, frames_per_video, frame_interval)
            
            frames_saved_this_video = 0
            
            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    continue
                
                # Run MegaDetector
                results = model.single_image_detection(frame)
                labels = results.get("labels", [])
                scores = results.get("scores", [])
                
                # Check for animal detections above threshold
                has_animal = False
                if len(labels) > 0:
                    if "animal" in labels[0] and float(labels[0][-4:]) >= confidence_thresh:
                        has_animal = True

                # Save frame if animal detected
                if has_animal:
                    save_frame_to_disk(frame, split_output_dir, video_path.stem, frame_idx)
                    frames_saved_this_video += 1
                    total_frames_saved += 1
            
            cap.release()
            
            if frames_saved_this_video > 0:
                tqdm.write(f"  ✅ {video_name}: {frames_saved_this_video} frames saved to {split}")
    
    print(f"\n🎉 Processing complete! Total bird frames saved: {total_frames_saved}")

# --------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract animal frames from bird-labeled videos using MegaDetector")
    parser.add_argument("--csv_path", type=str, default="test.csv", 
                       help="CSV file with video filename and species labels")
    parser.add_argument("--videos_dir", type=str, default="dataset/test",
                       help="Directory containing test videos")
    parser.add_argument("--output_dir", type=str, default="Kaggle/cupybara/yolo11_dataset_filtered",
                       help="Output directory for YOLO dataset")
    parser.add_argument("--confidence", type=float, default=0.70,
                       help="Confidence threshold for animal detection")
    parser.add_argument("--frames_per_video", type=int, default=32,
                       help="Maximum frames to extract per video")
    parser.add_argument("--frame_interval", type=int, default=4,
                       help="Sample every N frames for short videos")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible video splitting")
    
    args = parser.parse_args()
    
    # Validate inputs
    csv_path = Path(args.csv_path)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    # Read bird videos from CSV
    print("📋 Reading CSV for bird-labeled videos...")
    bird_videos = read_csv_for_species(csv_path, "bird")
    print(f"📹 Found {len(bird_videos)} videos labeled as 'bird'")
    
    if not bird_videos:
        print("❌ No bird videos found in CSV. Exiting.")
        return
    
    # Split videos into train/val/test
    rng = random.Random(args.seed)
    video_splits = split_videos_80_10_10(bird_videos, rng)
    
    print(f"📊 Video splits: train={len(video_splits['train'])}, "
          f"val={len(video_splits['val'])}, test={len(video_splits['test'])}")
    
    # Load MegaDetector model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    
    print(f"🔥 Loading MegaDetector v6 on {device}...")
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    print("✅ Model loaded successfully")
    
    # Process videos
    process_bird_videos(
        video_splits,
        videos_dir,
        output_dir,
        model,
        confidence_thresh=args.confidence,
        frames_per_video=args.frames_per_video,
        frame_interval=args.frame_interval
    )


if __name__ == "__main__":
    main() 