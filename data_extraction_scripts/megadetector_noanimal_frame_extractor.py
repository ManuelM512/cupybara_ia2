#!/usr/bin/env python3
"""
megadetector_noanimal_frame_extractor.py
=======================================
Extract *empty* frames (no animal) using MegaDetector v6
-------------------------------------------------------
This script is the negative counterpart of ``megadetector_video_frame_extractor.py``.

It reads *test.csv*, keeps only rows whose ``Species`` column is **no_animal**, and
processes the corresponding videos under ``dataset/test``.  Frames in which
MegaDetector **does not** detect the *animal* class are written to the existing
YOLO-style dataset ``Kaggle/cupybara/yolo11_dataset_filtered`` using the same
80 / 10 / 10 video-level split:

    <dataset_root>/
        train/no_animal/*.jpg   # 80 % of the videos
        val/no_animal/*.jpg     # 10 %
        test/no_animal/*.jpg    # 10 %

Frames originating from one video are never placed in different splits.

Example
-------
```bash
python megadetector_noanimal_frame_extractor.py \
        --csv_path test.csv \
        --videos_dir dataset/test \
        --output_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --confidence 0.25 \
        --frames_per_video 30 \
        --frame_interval 30
```
Dependencies (already in *requirements.txt*):
    • opencv-python
    • numpy
    • torch
    • ultralytics
    • tqdm
    • pytorch-wildlife  (provides the MegaDetectorV6 wrapper)
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
import torch
from PytorchWildlife.models import detection as pw_detection
import yaml

# --------------------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------------------

def read_csv_records(csv_path: Path) -> List[Tuple[str, str]]:
    """Return list of (filename, species) tuples from *csv_path*."""
    records: List[Tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            fname = row[0].strip().strip("'\"")
            species = row[1].strip().strip("'\"") if len(row) > 1 else "unknown"
            records.append((fname, species))
    return records

# --------------------------------------------------------------------------------------
# Split helper – guarantees non-empty val & test when possible
# --------------------------------------------------------------------------------------

def split_videos(videos: List[str], rng: random.Random) -> Dict[str, List[str]]:
    """Return dict with keys train / val / test using 80-10-10 rule.

    Ensures *val* and *test* are non-empty when the total video count ≥ 3.
    """
    rng.shuffle(videos)
    n = len(videos)

    if n == 0:
        return {"train": [], "val": [], "test": []}

    if n < 3:
        # Fallback – keep at least one video for validation when we have 2 videos
        train_cut = max(1, n - 1)
        val_cut = n  # rest goes to val
        return {
            "train": videos[:train_cut],
            "val": videos[train_cut:val_cut],
            "test": [],
        }

    train_cut = int(round(n * 0.8))
    val_cut = train_cut + int(round(n * 0.1))
    # Guarantee at least 1 in val and test
    if train_cut == 0:
        train_cut = 1
    if val_cut == train_cut:
        val_cut += 1
    if val_cut >= n:
        val_cut = n - 1
    return {
        "train": videos[:train_cut],
        "val": videos[train_cut:val_cut],
        "test": videos[val_cut:],
    }

# --------------------------------------------------------------------------------------
# Frame utilities
# --------------------------------------------------------------------------------------

def select_frame_indices(total_frames: int, *, frames_per_video: int, frame_interval: int) -> List[int]:
    """Return list of frame indices to sample from video."""
    if total_frames <= 0:
        return []

    if total_frames <= frames_per_video:
        return list(range(0, total_frames, max(1, frame_interval)))
    return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()


def save_frame(frame: np.ndarray, dst_dir: Path, video_stem: str, frame_idx: int):
    dst_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(dst_dir / filename), frame)

# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------

def process_noanimal_videos(
    video_names_by_split: Dict[str, List[str]],
    videos_dir: Path,
    output_dir: Path,
    model: pw_detection.MegaDetectorV6,
    *,
    confidence_thresh: float,
    frames_per_video: int,
    frame_interval: int,
):
    """Iterate videos and save frames that have **no** animal detections."""
    saved_total = 0

    for split, video_names in video_names_by_split.items():
        if not video_names:
            continue
        dst_base = output_dir / split / "no_animal"

        for vid_name in tqdm(video_names, desc=f"{split} videos", leave=False):
            vid_path = videos_dir / f"{vid_name}.mp4"
            if not vid_path.exists():
                # try any extension
                alt = next(vid_path.parent.glob(f"{vid_name}.*"), None)
                if alt is None:
                    tqdm.write(f"⚠️ Video not found: {vid_name}")
                    continue
                vid_path = alt

            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                tqdm.write(f"⚠️ Cannot open {vid_path.name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = select_frame_indices(total_frames, frames_per_video=frames_per_video, frame_interval=frame_interval)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                result = model.single_image_detection(frame)
                labels = result.get("labels", [])
                scores = result.get("scores", [])

                # Determine if any *animal* detection above threshold exists
                has_animal = False
                for lbl, sc in zip(labels, scores):
                    if lbl.startswith("animal") and float(sc) >= confidence_thresh:
                        has_animal = True
                        break

                if not has_animal:
                    save_frame(frame, dst_base, vid_path.stem, idx)
                    saved_total += 1

            cap.release()

    print("\n✅ Done – frames without animals saved:", saved_total)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract *no-animal* frames from videos using MegaDetector v6.")
    parser.add_argument("--csv_path", type=str, default="test.csv", help="CSV with Filename,Species columns")
    parser.add_argument("--videos_dir", type=str, default="dataset/test", help="Directory with test videos (*.mp4)")
    parser.add_argument("--output_dir", type=str, default="Kaggle/cupybara/yolo11_dataset_filtered", help="YOLO dataset root")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold for *animal* detection")
    parser.add_argument("--frames_per_video", type=int, default=2, help="Max frames sampled per video")
    parser.add_argument("--frame_interval", type=int, default=2, help="Sample every Nth frame when total < frames_per_video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not videos_dir.exists():
        raise FileNotFoundError(videos_dir)

    # ------------------------------------------------------------------
    # Filter CSV for no_animal videos and build 80/10/10 split
    # ------------------------------------------------------------------
    records = read_csv_records(csv_path)
    noanimal_videos = [fname for fname, sp in records if sp == "no_animal"]
    rng = random.Random(args.seed)
    split_map = split_videos(noanimal_videos, rng)

    # Ensure dataset folders exist (train/val/test/no_animal)
    for split in ["train", "val", "test"]:
        (output_dir / split / "no_animal").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load MegaDetector v6
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    print(f"🔥 Loading MegaDetector v6 on {device} …")
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    print("✅ Model loaded")

    # ------------------------------------------------------------------
    # Process videos
    # ------------------------------------------------------------------
    process_noanimal_videos(
        split_map,
        videos_dir,
        output_dir,
        model,
        confidence_thresh=args.confidence,
        frames_per_video=args.frames_per_video,
        frame_interval=args.frame_interval,
    )

if __name__ == "__main__":
    main() 