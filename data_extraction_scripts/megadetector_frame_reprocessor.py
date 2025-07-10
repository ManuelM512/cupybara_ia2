#!/usr/bin/env python3
"""
MegaDetector Frame Re-processor

This utility walks through an existing YOLO-style dataset folder (expected structure::

    <dataset_root>/
        train/<species>/*.jpg
        val/<species>/*.jpg
        test/<species>/*.jpg

and re-runs every image through the Microsoft AI-for-Earth **MegaDetector v6** model.  Only
frames that still contain at least one *animal* detection (MegaDetector class id 2) are
copied to a new dataset directory that mirrors the original split/species structure.

Typical usage – re-process the frames currently in ``Kaggle/cupybara/yolo11_dataset`` and
write the filtered result to ``Kaggle/cupybara/yolo11_dataset_filtered``::

    python megadetector_frame_reprocessor.py \
        --input_dir Kaggle/cupybara/yolo11_dataset \
        --output_dir Kaggle/cupybara/yolo11_dataset_filtered \
        --model_path megadetector_v5.pt \
        --confidence 0.25

If you'd rather *overwrite* the existing dataset in-place, pass the same path for
``--input_dir`` and ``--output_dir`` (frames with no animal detection will be **deleted**).

Dependencies (already listed in requirements.txt):
    • ultralytics>=8.1.0
    • opencv-python
    • numpy
    • tqdm
    • PyYAML
    • pytorch-wildlife  (provides the MegaDetectorV6 wrapper)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
import torch
from PytorchWildlife.models import detection as pw_detection

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def list_image_files(directory: Path) -> List[Path]:
    """Recursively list all *.jpg / *.png files under *directory*."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def ensure_output_structure(output_root: Path, sub_path: Path):
    """Create parent folders for *output_root / sub_path* if they don't exist."""
    target_dir = output_root / sub_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Main re-processing routine
# --------------------------------------------------------------------------------------

def reprocess_frames(
    input_root: Path,
    output_root: Path,
    model: pw_detection.MegaDetectorV6,
    *,
    confidence_thresh: float = 0.25,
    delete_empty: bool = False,
):
    """Run MegaDetector on all frames and copy/filter them to *output_root*.

    If *output_root* is the same as *input_root* and *delete_empty* is True, frames with
    no animal detection will be removed; otherwise they are left untouched.
    """
    all_images = list_image_files(input_root)
    if not all_images:
        print(f"No images found under {input_root}")
        return

    kept, removed = 0, 0
    for img_path in tqdm(all_images, desc="Re-processing frames"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Inference
        results = model.single_image_detection(img)
        labels = results.get("labels", [])
        scores = results.get("scores", [])

        has_animal = False
        if len(labels) > 0:
            if labels[0][:6] == "animal" and float(labels[0][-4:]) >= confidence_thresh:
                has_animal = True

        rel_path = img_path.relative_to(input_root)
        if has_animal:
            if output_root == input_root:
                # Keep as-is
                kept += 1
            else:
                # Copy to mirrored location
                ensure_output_structure(output_root, rel_path)
                shutil.copy2(img_path, output_root / rel_path)
                kept += 1
        else:
            # No animal → optionally delete / skip copying
            removed += 1
            if output_root == input_root and delete_empty:
                img_path.unlink(missing_ok=True)

    print("\n✅ Re-processing complete")
    print(f"Frames kept : {kept}")
    print(f"Frames removed/skipped: {removed}")

# --------------------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-process existing frames using MegaDetector.")
    parser.add_argument("--input_dir", type=str, default="Kaggle/cupybara/yolo11_dataset", help="Dataset root with train/val/test sub-folders")
    parser.add_argument("--output_dir", type=str, default="Kaggle/cupybara/yolo11_dataset_filtered", help="Destination dataset root (can be same as input_dir)")
    parser.add_argument("--model_path", type=str, default="megadetector_v5.pt", help="Path to MegaDetector weight file (.pt) – not used when loading pretrained MDv6")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold for animal detection")
    parser.add_argument("--delete_empty", action="store_true", help="If output_dir == input_dir, delete frames without animals instead of leaving them")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    # Load MegaDetector model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    print(f"🔥 Loading MegaDetector v6 on {device} …")
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    print("✅ Model ready")

    reprocess_frames(
        input_root,
        output_root,
        model,
        confidence_thresh=args.confidence,
        delete_empty=args.delete_empty,
    )


if __name__ == "__main__":
    main() 