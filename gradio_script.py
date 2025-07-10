#!/usr/bin/env python3
"""
gradio_video_classifier_ensemble_v2.py
======================================
Ensemble video classification using existing SpeciesNet extraction functionality.

This version uses the existing extract_crops_with_speciesnet.py functionality
instead of requiring the speciesnet module installation.

Pipeline:
1. Extract frames from video
2. Use existing SpeciesNet detection to find animals  
3. Crop detected animals
4. Classify crops with trained model
5. Aggregate results

Usage:
    python gradio_video_classifier_ensemble_v2.py --model_path runs/filtered_train_augmented/exp3/weights/best.pt
"""

import argparse
import logging
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Import functionality from existing SpeciesNet script
import sys
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeciesNetDetector:
    """Simplified SpeciesNet detector using existing functionality."""
    
    def __init__(self, model_dir: str = "speciesnet-pytorch-v4.0.1a-v1", confidence_threshold: float = 0.1):
        """Initialize the detector."""
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        
        # Try to load MegaDetector for animal detection
        self.detector = None
        self._load_detector()
        
    def _load_detector(self):
        """Load animal detector."""
        try:
            # Try different MegaDetector model names
            detector_options = [
                "MDV6-yolov10n.pt",  # Your local file
                "md_v5a.0.0.pt",
                "yolov8n.pt"  # Fallback
            ]
            
            for detector_name in detector_options:
                detector_path = Path(detector_name)
                if detector_path.exists():
                    logger.info(f"Loading detector: {detector_path}")
                    self.detector = YOLO(str(detector_path))
                    logger.info("✅ Animal detector loaded successfully")
                    return
            
            logger.warning("No MegaDetector found, will use fallback detection")
            
        except Exception as e:
            logger.warning(f"Could not load detector: {e}")
            
    def detect_animals_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect animals in a frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if self.detector is not None:
            try:
                # Run detection
                results = self.detector(frame, conf=self.confidence_threshold, verbose=False)
                
                height, width = frame.shape[:2]
                
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.data.cpu().numpy()
                        
                        for box in boxes:
                            x1, y1, x2, y2, conf, cls = box
                            
                            # Convert to normalized coordinates [x_min, y_min, width, height]
                            x_min_norm = x1 / width
                            y_min_norm = y1 / height
                            width_norm = (x2 - x1) / width
                            height_norm = (y2 - y1) / height
                            
                            detection = {
                                'bbox': [float(x_min_norm), float(y_min_norm), float(width_norm), float(height_norm)],
                                'conf': float(conf),
                                'category': '1',  # Animal category
                                'class_id': int(cls)
                            }
                            detections.append(detection)
                            
            except Exception as e:
                logger.warning(f"Detection failed for frame: {e}")
        
        # Fallback: create full-frame detection if no animals detected
        if not detections:
            detections.append({
                'bbox': [0.0, 0.0, 1.0, 1.0],  # Full frame
                'conf': 0.5,
                'category': '1',
                'class_id': 0
            })
        
        return detections


class EnsembleVideoClassifierV2:
    """Ensemble video classifier using existing SpeciesNet functionality."""
    
    def __init__(self, 
                 classifier_model_path: str,
                 confidence_threshold: float = 0.25,
                 detection_threshold: float = 0.1,
                 speciesnet_model_dir: str = "speciesnet-pytorch-v4.0.1a-v1"):
        """
        Initialize the ensemble classifier.
        
        Args:
            classifier_model_path: Path to trained YOLO classification model
            confidence_threshold: Minimum confidence for classification
            detection_threshold: Minimum confidence for detection
            speciesnet_model_dir: Path to SpeciesNet model directory
        """
        self.classifier_model_path = Path(classifier_model_path)
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold
        
        # Load classification model
        logger.info(f"Loading classification model from {self.classifier_model_path}")
        self.classifier = YOLO(str(self.classifier_model_path))
        
        # Get class names
        if hasattr(self.classifier, 'names') and self.classifier.names is not None:
            self.class_names = list(self.classifier.names.values())
        elif hasattr(self.classifier.model, 'names') and self.classifier.model.names is not None:
            self.class_names = list(self.classifier.model.names.values())
        else:
            self.class_names = [f"class_{i}" for i in range(14)]
        
        logger.info(f"Classification model loaded with {len(self.class_names)} classes: {self.class_names}")
        
        # Initialize detector
        self.detector = SpeciesNetDetector(
            model_dir=speciesnet_model_dir,
            confidence_threshold=detection_threshold
        )
    
    def convert_video_for_web(self, video_path: str) -> str:
        """
        Convert video to web-compatible format for Gradio display.
        
        Args:
            video_path: Original video path
            
        Returns:
            Path to converted video
        """
        try:
            import tempfile
            import subprocess
            
            # Create temporary file for converted video
            temp_dir = Path(tempfile.gettempdir())
            converted_path = temp_dir / f"converted_{Path(video_path).stem}.mp4"
            
            # Check if ffmpeg is available
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                ffmpeg_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                ffmpeg_available = False
                logger.warning("FFmpeg not available, using OpenCV for conversion")
            
            if ffmpeg_available:
                # Use FFmpeg for better conversion
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-c:v", "libx264",  # H.264 codec
                    "-preset", "fast",   # Fast encoding
                    "-crf", "28",        # Reasonable quality/size
                    "-c:a", "aac",       # AAC audio
                    "-movflags", "+faststart",  # Web optimization
                    "-y",                # Overwrite output
                    str(converted_path)
                ]
                
                logger.info("Converting video to web-compatible format...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and converted_path.exists():
                    logger.info(f"✅ Video converted successfully: {converted_path}")
                    return str(converted_path)
                else:
                    logger.warning(f"FFmpeg conversion failed: {result.stderr}")
            
            # Fallback: Use OpenCV for conversion
            logger.info("Using OpenCV for video conversion...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.warning("Could not open video for conversion")
                return video_path
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create VideoWriter for web-compatible format
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(str(converted_path), fourcc, fps, (width, height))
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
                # Limit conversion to reasonable length
                if frame_count > 1000:  # ~33 seconds at 30fps
                    break
            
            cap.release()
            out.release()
            
            if converted_path.exists():
                logger.info(f"✅ Video converted with OpenCV: {converted_path}")
                return str(converted_path)
            else:
                logger.warning("OpenCV conversion failed")
                return video_path
                
        except Exception as e:
            logger.warning(f"Video conversion failed: {e}")
            return video_path
    
    def extract_frames(self, video_path: str, max_frames: int = 100, frame_interval: int = 1) -> List[np.ndarray]:
        """Extract frames from video."""
        logger.info(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")
        return frames
    
    def detect_and_crop_animals(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, Dict]]:
        """
        Detect animals in frames and extract crops.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of (crop_image, detection_info) tuples
        """
        logger.info(f"Detecting animals in {len(frames)} frames")
        
        crops = []
        
        for frame_idx, frame in enumerate(tqdm(frames, desc="Detecting animals")):
            height, width = frame.shape[:2]
            
            # Detect animals in frame
            detections = self.detector.detect_animals_in_frame(frame)
            
            for detection in detections:
                confidence = detection.get('conf', 0.0)
                
                # Filter by confidence
                if confidence < self.detection_threshold:
                    continue
                
                # Extract bbox
                bbox = detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x_min_norm, y_min_norm, width_norm, height_norm = bbox
                
                x_min = int(x_min_norm * width)
                y_min = int(y_min_norm * height)
                x_max = int((x_min_norm + width_norm) * width)
                y_max = int((y_min_norm + height_norm) * height)
                
                # Ensure coordinates are within bounds
                x_min = max(0, min(x_min, width - 1))
                y_min = max(0, min(y_min, height - 1))
                x_max = max(x_min + 1, min(x_max, width))
                y_max = max(y_min + 1, min(y_max, height))
                
                # Extract crop
                crop = frame[y_min:y_max, x_min:x_max]
                
                if crop.size > 0:
                    detection_info = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'frame_idx': frame_idx,
                        'crop_coords': (x_min, y_min, x_max, y_max)
                    }
                    crops.append((crop, detection_info))
        
        logger.info(f"Extracted {len(crops)} animal crops")
        return crops
    
    def classify_crops(self, crops: List[Tuple[np.ndarray, Dict]]) -> List[Dict]:
        """Classify extracted crops."""
        logger.info(f"Classifying {len(crops)} crops")
        
        results = []
        
        for crop, detection_info in tqdm(crops, desc="Classifying crops"):
            try:
                # Run classification
                classification_results = self.classifier(crop, verbose=False)
                
                for result in classification_results:
                    if hasattr(result, 'probs') and result.probs is not None:
                        confidences = result.probs.data.cpu().numpy()
                        
                        # Get all class predictions above threshold
                        predictions = []
                        for class_id, conf in enumerate(confidences):
                            if conf >= self.confidence_threshold:
                                class_name = self.class_names[class_id]
                                predictions.append({
                                    'class': class_name,
                                    'confidence': float(conf),
                                    'class_id': class_id
                                })
                        
                        # Sort by confidence
                        predictions.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        # Store result
                        classification_result = {
                            'detection_info': detection_info,
                            'predictions': predictions,
                            'top_prediction': predictions[0] if predictions else None
                        }
                        results.append(classification_result)
                        
            except Exception as e:
                logger.warning(f"Failed to classify crop: {e}")
                continue
        
        logger.info(f"Successfully classified {len(results)} crops")
        return results
    
    def aggregate_results(self, classification_results: List[Dict]) -> Dict:
        """Aggregate classification results."""
        # Collect all predictions
        all_predictions = defaultdict(list)
        detection_counts = defaultdict(int)
        total_detections = len(classification_results)
        
        for result in classification_results:
            top_pred = result.get('top_prediction')
            if top_pred:
                class_name = top_pred['class']
                confidence = top_pred['confidence']
                
                all_predictions[class_name].append(confidence)
                detection_counts[class_name] += 1
        
        # Calculate aggregated metrics
        final_scores = {}
        for class_name in self.class_names:
            confidences = all_predictions[class_name]
            
            if confidences:
                avg_confidence = np.mean(confidences)
                max_confidence = np.max(confidences)
                detection_frequency = len(confidences) / total_detections if total_detections > 0 else 0
                
                # Weighted final score
                final_score = (avg_confidence * 0.6 + max_confidence * 0.2 + detection_frequency * 0.2)
            else:
                avg_confidence = 0.0
                max_confidence = 0.0
                detection_frequency = 0.0
                final_score = 0.0
            
            final_scores[class_name] = {
                'final_score': final_score,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'detection_frequency': detection_frequency,
                'detection_count': detection_counts[class_name]
            }
        
        # Sort by final score
        sorted_classes = sorted(final_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)
        
        return {
            'predictions': sorted_classes,
            'total_detections': total_detections,
            'unique_classes_detected': len([c for c, scores in final_scores.items() if scores['detection_count'] > 0]),
            'final_scores': final_scores,
            'raw_results': classification_results
        }
    
    def create_results_plot(self, aggregated_results: Dict) -> Figure:
        """Create visualization of results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top predictions
        predictions = aggregated_results['predictions'][:10]
        classes = [pred[0] for pred in predictions]
        scores = [pred[1]['final_score'] for pred in predictions]
        
        ax1.barh(classes, scores, color='skyblue')
        ax1.set_xlabel('Final Score')
        ax1.set_title('Top Predictions (Ensemble Score)')
        ax1.invert_yaxis()
        
        # Detection counts
        detection_counts = [pred[1]['detection_count'] for pred in predictions]
        ax2.bar(classes, detection_counts, color='lightcoral')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title('Detection Counts by Class')
        ax2.tick_params(axis='x', rotation=45)
        
        # Confidence vs count scatter
        confidences = [pred[1]['avg_confidence'] for pred in predictions]
        ax3.scatter(detection_counts, confidences, s=[score*100 for score in scores], alpha=0.6)
        ax3.set_xlabel('Detection Count')
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Detections vs Confidence')
        
        # Top 5 pie chart
        frequencies = [pred[1]['detection_frequency'] for pred in predictions[:5]]
        if sum(frequencies) > 0:
            ax4.pie(frequencies, labels=classes[:5], autopct='%1.1f%%', startangle=90)
            ax4.set_title('Top 5 Classes by Detection Frequency')
        else:
            ax4.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('No detections found')
        
        plt.tight_layout()
        return fig
    
    def classify_video_ensemble(
        self, 
        video_path: str, 
        max_frames: int = 100, 
        frame_interval: int = 1,
        progress_callback: Optional = None
    ) -> Tuple[str, Dict, Figure]:
        """Run ensemble classification on video."""
        try:
            # Step 1: Extract frames
            if progress_callback:
                progress_callback(0.1)
            
            frames = self.extract_frames(video_path, max_frames, frame_interval)
            
            if not frames:
                return "No frames could be extracted from video", {}, None
            
            # Step 2: Detect animals and extract crops
            if progress_callback:
                progress_callback(0.4)
            
            crops = self.detect_and_crop_animals(frames)
            
            if not crops:
                return "No animals detected in video frames", {"total_detections": 0}, None
            
            # Step 3: Classify crops
            if progress_callback:
                progress_callback(0.7)
            
            classification_results = self.classify_crops(crops)
            
            # Step 4: Aggregate results
            if progress_callback:
                progress_callback(0.9)
            
            aggregated = self.aggregate_results(classification_results)
            
            # Create result text
            if not aggregated['predictions']:
                return "No confident predictions from ensemble", aggregated, None
            
            # Generate summary text
            top_predictions = aggregated['predictions'][:5]
            result_text = f"**Ensemble Video Classification Results**\n\n"
            result_text += f"🎯 **Total Detections**: {aggregated['total_detections']}\n"
            result_text += f"🔍 **Unique Classes**: {aggregated['unique_classes_detected']}\n\n"
            result_text += "**Top Predictions:**\n"
            
            for i, (class_name, scores) in enumerate(top_predictions, 1):
                if scores['final_score'] > 0:
                    result_text += f"{i}. **{class_name}** - Score: {scores['final_score']:.3f} "
                    result_text += f"(Detections: {scores['detection_count']}, "
                    result_text += f"Avg Conf: {scores['avg_confidence']:.3f})\n"
            
            # Create plot
            if progress_callback:
                progress_callback(1.0)
            
            plot = self.create_results_plot(aggregated)
            
            return result_text, aggregated, plot
            
        except Exception as e:
            logger.error(f"Error in ensemble classification: {e}")
            return f"Error during classification: {str(e)}", {}, None


def create_gradio_interface(model_path: str, confidence_threshold: float = 0.25, detection_threshold: float = 0.1):
    """Create Gradio interface."""
    
    classifier = EnsembleVideoClassifierV2(
        classifier_model_path=model_path,
        confidence_threshold=confidence_threshold,
        detection_threshold=detection_threshold
    )
    
    def process_video(video_file, max_frames, frame_interval, conf_threshold, det_threshold):
        if video_file is None:
            return "Please upload a video file", None, None, None
        
        # Update thresholds
        classifier.confidence_threshold = conf_threshold
        classifier.detection_threshold = det_threshold
        
        # Convert video for web display
        logger.info("Converting video for web display...")
        converted_video_path = classifier.convert_video_for_web(video_file)
        
        # Process video for classification (use original)
        result_text, result_json, result_plot = classifier.classify_video_ensemble(
            video_file, 
            max_frames, 
            frame_interval
        )
        
        return result_text, result_json, result_plot, converted_video_path
    
    # Create interface
    with gr.Blocks(title="Ensemble Video Animal Classifier V2") as demo:
        gr.Markdown("# 🎥 Ensemble Video Animal Classifier V2")
        gr.Markdown("Upload a video to classify animals using detection + classification ensemble (no SpeciesNet module required)")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                
                with gr.Accordion("Processing Settings", open=False):
                    max_frames = gr.Slider(1, 500, value=100, step=1, label="Max Frames")
                    frame_interval = gr.Slider(1, 10, value=1, step=1, label="Frame Interval")
                    conf_threshold = gr.Slider(0.01, 1.0, value=confidence_threshold, step=0.01, label="Classification Confidence")
                    det_threshold = gr.Slider(0.01, 1.0, value=detection_threshold, step=0.01, label="Detection Confidence")
                
                classify_btn = gr.Button("🔍 Classify Video", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                result_text = gr.Markdown(label="Results")
                converted_video = gr.Video(label="Video Player (Web Compatible)", visible=False)
                result_plot = gr.Plot(label="Analysis")
                
                with gr.Accordion("Detailed Results", open=False):
                    result_json = gr.JSON(label="Raw Results")
        
        def update_interface(video_file, max_frames, frame_interval, conf_threshold, det_threshold):
            result_text, result_json, result_plot, converted_video_path = process_video(
                video_file, max_frames, frame_interval, conf_threshold, det_threshold
            )
            
            # Show converted video if available
            if converted_video_path and converted_video_path != video_file:
                return result_text, result_json, result_plot, gr.Video(value=converted_video_path, visible=True)
            else:
                return result_text, result_json, result_plot, gr.Video(visible=False)
        
        classify_btn.click(
            fn=update_interface,
            inputs=[video_input, max_frames, frame_interval, conf_threshold, det_threshold],
            outputs=[result_text, result_json, result_plot, converted_video]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Ensemble Video Animal Classifier V2")
    parser.add_argument("--model_path", type=str, default="runs/crops_to_frames/crops_training/weights/best.pt", help="Path to trained classification model")
    parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Classification confidence threshold")
    parser.add_argument("--detection_threshold", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7863, help="Server port")
    
    args = parser.parse_args()
    
    logger.info(f"Using classification model: {args.model_path}")
    logger.info(f"Classification confidence threshold: {args.confidence_threshold}")
    logger.info(f"Detection confidence threshold: {args.detection_threshold}")
    
    demo = create_gradio_interface(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        detection_threshold=args.detection_threshold
    )
    
    logger.info(f"Starting Gradio server on {args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main() 