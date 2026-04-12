#!/usr/bin/env python3
"""
MOG2 vs YOLO Frame-by-Frame Accuracy Comparison

Compares MOG2 motion detection with YOLO object detection to calculate:
- Precision, Recall, F1-Score
- True Positives, False Positives, False Negatives
- IOU (Intersection over Union) per frame
- Overall accuracy metrics
"""

import cv2
import sys
import os
import argparse
import json
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.yolo_detector import YOLODetector
from detection.mog2_detector import MOG2OnlyDetector

# Settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
IOU_THRESHOLD = 0.3  # Minimum IOU to consider a match


class MetricsCalculator:
    """Calculate detection accuracy metrics."""
    
    @staticmethod
    def compute_iou(box1: List, box2: List) -> float:
        """Compute Intersection over Union (IOU) between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def match_detections(mog2_boxes: List, yolo_boxes: List, iou_threshold: float = 0.3) -> Tuple[int, int, int]:
        """
        Match MOG2 detections to YOLO detections using IOU.
        
        Args:
            mog2_boxes: List of MOG2 detections
            yolo_boxes: List of YOLO detections (ground truth)
            iou_threshold: Minimum IOU to consider a match
        
        Returns:
            (true_positives, false_positives, false_negatives)
        """
        matched_yolo = set()
        true_positives = 0
        
        # Match each MOG2 detection to best YOLO detection
        for mog2_box in mog2_boxes:
            best_iou = 0
            best_yolo_idx = -1
            
            for yolo_idx, yolo_box in enumerate(yolo_boxes):
                if yolo_idx in matched_yolo:
                    continue
                
                iou = MetricsCalculator.compute_iou(mog2_box, yolo_box)
                if iou > best_iou:
                    best_iou = iou
                    best_yolo_idx = yolo_idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_yolo.add(best_yolo_idx)
        
        false_positives = len(mog2_boxes) - true_positives
        false_negatives = len(yolo_boxes) - true_positives
        
        return true_positives, false_positives, false_negatives
    
    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int) -> Dict:
        """Calculate precision, recall, and F1-score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }


class MOG2YOLOComparator:
    """Compare MOG2 and YOLO detections frame by frame."""
    
    def __init__(self, video_path: str, calibrate: bool = False):
        """
        Initialize comparator.
        
        Args:
            video_path: Path to video file
            calibrate: Whether to calibrate MOG2 using YOLO
        """
        self.video_path = video_path
        self.calibrate = calibrate
        
        # Initialize detectors
        self.mog2_detector = MOG2OnlyDetector(history=500, var_threshold=16)
        self.yolo_detector = YOLODetector(model_name="yolov8n.pt")
        
        # Metrics
        self.frame_metrics = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
    
    def process_video(self, max_frames: int = None, show_frames: bool = False) -> Dict:
        """
        Process video and compare detections.
        
        Args:
            max_frames: Maximum frames to process (None = all)
            show_frames: Display frames during processing
        
        Returns:
            Dictionary with overall metrics
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video")
            return {}
        
        frame_count = 0
        processed_frames = 0
        
        print(f"\n{'='*70}")
        print(f"MOG2 vs YOLO Accuracy Comparison")
        print(f"{'='*70}")
        print(f"Video: {os.path.basename(self.video_path)}")
        print(f"Calibrate MOG2: {self.calibrate}")
        print(f"IOU Threshold: {IOU_THRESHOLD}")
        print(f"{'='*70}\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if max_frames and processed_frames >= max_frames:
                    break
                
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Get detections
                mog2_boxes = self.mog2_detector.get_motion_boxes(frame)
                yolo_boxes = self.yolo_detector.detect(frame)
                
                # Calibrate MOG2 if requested
                if self.calibrate and frame_count % 10 == 0:
                    self.mog2_detector.calibrate_from_yolo(frame, yolo_boxes, iou_threshold=0.2)
                
                # Calculate metrics for this frame
                tp, fp, fn = MetricsCalculator.match_detections(
                    mog2_boxes, yolo_boxes, 
                    iou_threshold=IOU_THRESHOLD
                )
                
                metrics = MetricsCalculator.calculate_metrics(tp, fp, fn)
                
                # Store frame metrics
                frame_data = {
                    "frame": frame_count,
                    "mog2_count": len(mog2_boxes),
                    "yolo_count": len(yolo_boxes),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "metrics": metrics
                }
                self.frame_metrics.append(frame_data)
                
                # Accumulate totals
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
                
                # Print progress
                processed_frames += 1
                if processed_frames % 30 == 0:
                    print(f"[Frame {frame_count}] MOG2: {len(mog2_boxes):2d} | YOLO: {len(yolo_boxes):2d} | "
                          f"TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d} | "
                          f"P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f} | F1: {metrics['f1_score']:.2f}")
                
                # Show frame
                if show_frames:
                    display_frame = self._draw_comparison(frame, mog2_boxes, yolo_boxes, metrics)
                    cv2.imshow("MOG2 vs YOLO Comparison", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            cap.release()
            if show_frames:
                cv2.destroyAllWindows()
        
        # Calculate overall metrics
        overall_metrics = MetricsCalculator.calculate_metrics(
            self.total_tp, self.total_fp, self.total_fn
        )
        
        return {
            "video": self.video_path,
            "frames_processed": processed_frames,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "overall_metrics": overall_metrics,
            "frame_metrics": self.frame_metrics
        }
    
    def _draw_comparison(self, frame: np.ndarray, mog2_boxes: List, yolo_boxes: List, 
                        metrics: Dict) -> np.ndarray:
        """Draw comparison visualization."""
        result = frame.copy()
        
        # Draw MOG2 boxes (RED)
        for box in mog2_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result, "MOG2", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw YOLO boxes (GREEN)
        for box in yolo_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, "YOLO", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display metrics
        cv2.putText(result, f"P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f} | F1: {metrics['f1_score']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"MOG2: {len(mog2_boxes)} | YOLO: {len(yolo_boxes)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result


def print_results(results: Dict):
    """Print detailed results."""
    if not results:
        return
    
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY RESULTS")
    print(f"{'='*70}")
    print(f"Video: {os.path.basename(results['video'])}")
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"\nTotal Detections:")
    print(f"  True Positives:  {results['total_tp']}")
    print(f"  False Positives: {results['total_fp']}")
    print(f"  False Negatives: {results['total_fn']}")
    print(f"\nOverall Metrics:")
    metrics = results['overall_metrics']
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"{'='*70}\n")


def save_results(results: Dict, output_file: str = "mog2_yolo_comparison.json"):
    """Save detailed results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MOG2 vs YOLO Accuracy Comparison")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate MOG2 using YOLO")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--show", action="store_true", help="Show frame-by-frame comparison")
    parser.add_argument("--save", type=str, default="mog2_yolo_comparison.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Run comparison
    comparator = MOG2YOLOComparator(video_path, calibrate=args.calibrate)
    results = comparator.process_video(max_frames=args.max_frames, show_frames=args.show)
    
    # Print results
    print_results(results)
    
    # Save results
    if args.save:
        save_results(results, args.save)


if __name__ == "__main__":
    main()
