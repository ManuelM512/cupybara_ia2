#!/usr/bin/env python3
"""
MegaDetector Video Frame Extractor

This script reads a CSV file (`test.csv`) that maps video filenames to species labels, processes videos from the
`dataset/test` folder, and uses the Microsoft AI-for-Earth *MegaDetector* model to detect the presence of animals in
each frame.

If a frame contains at least one detection of category **animal** (class id 2 in MegaDetector), the frame is saved as an
image to the `Kaggle/cupybara/yolo11_dataset` folder with the following split distribution **per species**:

 • 80 % of the videos → `train/`
 • 10 % of the videos → `val/`
 • 10 % of the videos → `test/`

All frames that originate from the same video are stored together in the same split sub-folder to avoid data leakage
between training and validation/testing sets.

Usage (default paths work out-of-the-box):

    python megadetector_video_frame_extractor.py \
        --csv_path test.csv \
        --videos_dir dataset/test \
        --output_dir Kaggle/cupybara/yolo11_dataset \
        --model_path megadetector_v5.pt \
        --confidence 0.25 \
        --frames_per_video 30 \
        --frame_interval 15

The model weight file (`megadetector_v5.pt`) can be downloaded from the official MegaDetector release page:
https://github.com/microsoft/CameraTraps/releases

Dependencies: ultralytics>=8.1.0, opencv-python, pandas, numpy, tqdm, PyYAML (already listed in requirements.txt).
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def read_video_label_csv(csv_path: Path) -> List[Tuple[str, str]]:
    """Read the CSV (filename,species) and return a list of tuples."""
    records: List[Tuple[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row if present
        for row in reader:
            # Expected format: Filename,Species   (values may contain quotes)
            if not row:
                continue
            filename = row[0].strip().strip("'\"")
            species = row[1].strip().strip("'\"") if len(row) > 1 else "unknown"
            records.append((filename, species))
    return records


def split_videos_per_species(records: List[Tuple[str, str]], rng: random.Random) -> Dict[str, Dict[str, List[str]]]:
    """Split videos for every species into train/val/test list according to 80/10/10 (video-level).

    Returns a nested dict: {species: {split: [filename, ...], ...}, ...}
    """
    species_to_videos: Dict[str, List[str]] = defaultdict(list)
    for filename, species in records:
        species_to_videos[species].append(filename)

    species_to_split: Dict[str, Dict[str, List[str]]] = {}
    for species, videos in species_to_videos.items():
        rng.shuffle(videos)
        n = len(videos)
        if n == 0:
            continue
        train_cut = int(round(n * 0.8))
        val_cut = train_cut + int(round(n * 0.1))
        # Ensure at least one video in test if possible
        if val_cut >= n and n >= 2:
            val_cut = n - 1
        species_to_split[species] = {
            "train": videos[:train_cut],
            "val": videos[train_cut:val_cut],
            "test": videos[val_cut:],
        }
    return species_to_split


def ensure_output_dirs(root: Path, species_names: List[str]):
    """Create train/val/test/{species} sub-folders under `root`."""
    for split in ["train", "val", "test"]:
        for species in species_names:
            (root / split / species).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Frame extraction & saving helpers
# --------------------------------------------------------------------------------------

def extract_candidate_frames(cap: cv2.VideoCapture, total_frames: int, *, frames_per_video: int, frame_interval: int) -> List[int]:
    """Return a list of frame indices to sample from the video."""
    if total_frames <= 0:
        return []

    if total_frames <= frames_per_video:
        # Short video: sample every `frame_interval` frame or all if fewer
        return list(range(0, total_frames, max(1, frame_interval)))
    else:
        # Longer video: evenly spaced indices limited to frames_per_video
        return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()


def save_frame(frame: np.ndarray, dst_dir: Path, video_stem: str, frame_idx: int):
    """Write frame as JPEG to destination directory."""
    filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(dst_dir / filename), frame)


# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------

def process_videos(
    species_split_map: Dict[str, Dict[str, List[str]]],
    videos_dir: Path,
    output_dir: Path,
    model: pw_detection.MegaDetectorV6,
    *,
    confidence_thresh: float,
    frames_per_video: int,
    frame_interval: int,
):
    # Simple stats trackers
    stats_total_saved = 0
    species_saved: Dict[str, int] = defaultdict(int)

    # Iterate over species and splits
    for species, split_dict in species_split_map.items():
        for split, video_names in split_dict.items():
            dst_base_dir = output_dir / split / species
            if not video_names:
                continue

            for video_name in tqdm(video_names, desc=f"{species} ({split})", leave=False):
                video_path = videos_dir / f"{video_name}.mp4"
                if not video_path.exists():
                    # Try uppercase/lowercase variations or missing extension
                    video_path_alt = next(video_path.parent.glob(f"{video_name}.*"), None)
                    if video_path_alt is not None:
                        video_path = video_path_alt
                    else:
                        tqdm.write(f"⚠️ Video not found: {video_path}")
                        continue

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    tqdm.write(f"⚠️ Cannot open video {video_path.name}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                candidate_indices = extract_candidate_frames(
                    cap, total_frames, frames_per_video=frames_per_video, frame_interval=frame_interval
                )

                for frame_idx in candidate_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()
                    if not success or frame is None:
                        continue

                    # Run MegaDetector inference
                    results = model.single_image_detection(frame)
                    if "animal" in str(results["labels"]):
                        save_frame(frame, dst_base_dir, video_path.stem, frame_idx)
                        stats_total_saved += 1
                        species_saved[species] += 1

                cap.release()

    # Print brief summary
    print("\n✅ Processing complete")
    print(f"Total frames saved: {stats_total_saved}")
    for sp, count in species_saved.items():
        print(f"  {sp}: {count} frames")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract animal frames from videos using MegaDetector.")
    parser.add_argument("--csv_path", type=str, default="test.csv", help="Path to CSV file with filename,species labels")
    parser.add_argument("--videos_dir", type=str, default="dataset/test", help="Directory containing source videos")
    parser.add_argument("--output_dir", type=str, default="Kaggle/cupybara/yolo11_dataset", help="Root output dataset directory")
    parser.add_argument("--model_path", type=str, default="megadetector_v5.pt", help="Path to MegaDetector weight file (.pt)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for animal detection")
    parser.add_argument("--frames_per_video", type=int, default=30, help="Maximum frames sampled per video")
    parser.add_argument("--frame_interval", type=int, default=30, help="Sample every N frames for short videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    rng = random.Random(args.seed)

    # Read CSV and build splits
    records = read_video_label_csv(csv_path)
    species_split_map = split_videos_per_species(records, rng)

    # Ensure output directory structure matches dataset.yaml classes (if available)
    dataset_yaml_path = output_dir / "dataset.yaml"
    if dataset_yaml_path.exists():
        with dataset_yaml_path.open("r") as f:
            yaml_cfg = yaml.safe_load(f)
            species_list = yaml_cfg.get("names", [])
    else:
        # Fall back to species present in CSV
        species_list = list({sp for _, sp in records})

    ensure_output_dirs(output_dir, species_list)

    # Load MegaDetector model (YOLO format)
    print("🔥 Loading MegaDetector model …")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.cuda.set_device(0)
    model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-c")
    print("✅ Model loaded")

    # Process videos
    process_videos(
        species_split_map,
        videos_dir,
        output_dir,
        model,
        confidence_thresh=args.confidence,
        frames_per_video=args.frames_per_video,
        frame_interval=args.frame_interval,
    )


if __name__ == "__main__":
    main() 