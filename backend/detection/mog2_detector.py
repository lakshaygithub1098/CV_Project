"""
MOG2-only motion detector trained by YOLO validation.

This detector uses only background subtraction (MOG2) to detect moving objects.
It learns motion thresholds from YOLO detections to improve accuracy.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from yolo_detector import YOLODetector


class MOG2OnlyDetector:
    """
    Pure MOG2-based motion detection with YOLO-based accuracy calibration.
    """
    
    # Traffic density categories
    DENSITY_LIGHT = "LIGHT"
    DENSITY_MEDIUM = "MEDIUM"
    DENSITY_HEAVY = "HEAVY"
    
    LIGHT_THRESHOLD = 5       # 0-5 vehicles
    MEDIUM_THRESHOLD = 10     # 6-10 vehicles
    # >10 is HEAVY
    
    def __init__(self, history: int = 500, var_threshold: float = 16, divider_pos: int = 640):
        """
        Initialize MOG2 detector with road division support.
        
        Args:
            history: Number of frames for background model
            var_threshold: Variance threshold for foreground detection
            divider_pos: X-coordinate to divide roads (default: 640 for 1280px width)
        """
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        
        # Calibration parameters learned from YOLO
        self.min_area = 400  # Minimum motion blob size
        self.max_area = None  # Maximum motion blob size
        self.calibrated = False
        
        # Road division
        self.divider_pos = divider_pos
        
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get foreground motion mask using MOG2.
        
        Args:
            frame: Input frame
        
        Returns:
            Binary foreground mask
        """
        fg_mask = self.mog2.apply(frame)
        
        # Remove shadows (value 128 in MOG2 output)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        return fg_mask
    
    def get_motion_boxes(self, frame: np.ndarray) -> List[List[float]]:
        """
        Extract bounding boxes from motion detection.
        
        Args:
            frame: Input frame
        
        Returns:
            List of [x1, y1, x2, y2, confidence] boxes
        """
        if self.max_area is None:
            self.max_area = frame.shape[0] * frame.shape[1]
        
        fg_mask = self.get_foreground_mask(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Confidence based on area ratio
            frame_area = frame.shape[0] * frame.shape[1]
            confidence = min(area / (self.min_area * 10), 1.0)
            
            boxes.append([x1, y1, x2, y2, confidence])
        
        return boxes
    
    def calibrate_from_yolo(self, frame: np.ndarray, yolo_boxes: List, iou_threshold: float = 0.2):
        """
        Calibrate MOG2 thresholds using YOLO detections as ground truth.
        
        Args:
            frame: Input frame
            yolo_boxes: List of YOLO detections [x1, y1, x2, y2, conf]
            iou_threshold: IOU threshold for matching boxes
        """
        if len(yolo_boxes) == 0:
            return
        
        mog2_boxes = self.get_motion_boxes(frame)
        
        # Count matches
        matched = 0
        for yolo_box in yolo_boxes:
            for mog2_box in mog2_boxes:
                iou = self._compute_iou(yolo_box, mog2_box)
                if iou > iou_threshold:
                    matched += 1
                    break
        
        # Adjust thresholds if needed
        match_rate = matched / len(yolo_boxes) if yolo_boxes else 0
        
        if match_rate < 0.7:  # Less than 70% match rate
            self.min_area = max(200, self.min_area - 50)  # More sensitive
            print(f"[MOG2 CALIBRATION] Match rate: {match_rate:.1%}, decreasing min_area to {self.min_area}")
        elif match_rate > 0.95:  # More than 95% match rate
            self.min_area = min(800, self.min_area + 50)  # Less sensitive
            print(f"[MOG2 CALIBRATION] Match rate: {match_rate:.1%}, increasing min_area to {self.min_area}")
        
        self.calibrated = True
    
    def get_road_for_box(self, box: List) -> str:
        """
        Assign box to Road A or Road B based on centroid x-coordinate.
        
        Args:
            box: [x1, y1, x2, y2, ...]
        
        Returns:
            "Road A" or "Road B"
        """
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2.0
        return "Road A" if cx < self.divider_pos else "Road B"
    
    def calculate_density_category(self, vehicle_count: int) -> str:
        """
        Classify traffic density.
        
        Args:
            vehicle_count: Number of vehicles
        
        Returns:
            Density level: "LIGHT", "MEDIUM", or "HEAVY"
        """
        if vehicle_count <= self.LIGHT_THRESHOLD:
            return self.DENSITY_LIGHT
        elif vehicle_count <= self.MEDIUM_THRESHOLD:
            return self.DENSITY_MEDIUM
        else:
            return self.DENSITY_HEAVY
    
    def compute_road_statistics(self, mog2_boxes: List) -> Dict:
        """
        Compute congestion statistics for both roads.
        
        Args:
            mog2_boxes: List of MOG2 detections [x1, y1, x2, y2, conf]
        
        Returns:
            Dict with Road A and Road B statistics
        """
        road_a_boxes = []
        road_b_boxes = []
        
        for box in mog2_boxes:
            road = self.get_road_for_box(box)
            if road == "Road A":
                road_a_boxes.append(box)
            else:
                road_b_boxes.append(box)
        
        # Statistics for Road A
        road_a_count = len(road_a_boxes)
        road_a_density = self.calculate_density_category(road_a_count)
        
        # Statistics for Road B
        road_b_count = len(road_b_boxes)
        road_b_density = self.calculate_density_category(road_b_count)
        
        return {
            "Road A": {
                "vehicle_count": road_a_count,
                "density": road_a_density,
                "boxes": road_a_boxes
            },
            "Road B": {
                "vehicle_count": road_b_count,
                "density": road_b_density,
                "boxes": road_b_boxes
            },
            "total_vehicles": road_a_count + road_b_count
        }
    
    @staticmethod
    def _compute_iou(box1: List, box2: List) -> float:
        """Compute IOU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


class MOG2Analyzer:
    """Visualization for MOG2-only detection with congestion analysis."""
    
    @staticmethod
    def get_density_color(density: str) -> Tuple[int, int, int]:
        """
        Get color based on density level.
        
        Args:
            density: Density level ("LIGHT", "MEDIUM", "HEAVY")
        
        Returns:
            BGR color tuple
        """
        if density == "LIGHT":
            return (0, 255, 0)  # GREEN
        elif density == "MEDIUM":
            return (0, 165, 255)  # ORANGE
        else:  # HEAVY
            return (0, 0, 255)  # RED
    
    @staticmethod
    def draw_mog2_frame(frame: np.ndarray, 
                       mog2_boxes: List,
                       road_stats: Dict = None,
                       yolo_boxes: List = None,
                       show_mask: bool = False,
                       motion_mask: np.ndarray = None) -> np.ndarray:
        """
        Draw MOG2 detections with road congestion analysis.
        
        Args:
            frame: Input frame
            mog2_boxes: MOG2 detections [x1, y1, x2, y2, conf]
            road_stats: Road statistics from compute_road_statistics()
            yolo_boxes: YOLO detections for comparison (optional)
            show_mask: Show motion mask overlay
            motion_mask: Binary motion mask from MOG2
        
        Returns:
            Annotated frame
        """
        result = frame.copy()
        frame_height = frame.shape[0]
        divider_pos = 640  # Assuming 1280 width
        
        # Draw road divider line
        if road_stats:
            divider_pos = int((road_stats.get("Road A", {}).get("boxes", []) or [640])[0][0]) if road_stats.get("Road A", {}).get("boxes") else 640
        cv2.line(result, (divider_pos, 0), (divider_pos, frame_height), (0, 255, 255), 2)
        
        # Draw MOG2 boxes in RED
        for box in mog2_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = box[4]
            
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)  # RED
            cv2.putText(
                result, f"MOG2 {conf*100:.0f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )
        
        # Draw YOLO boxes in GREEN for comparison
        if yolo_boxes:
            for box in yolo_boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)  # GREEN (thin)
                cv2.putText(
                    result, "YOLO",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )
        
        # Motion mask overlay
        if show_mask and motion_mask is not None:
            overlay = np.zeros_like(result)
            overlay[motion_mask > 0] = (255, 100, 0)  # Cyan overlay
            result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        # Draw road statistics
        if road_stats:
            # Road A statistics (left side)
            road_a_data = road_stats.get("Road A", {})
            road_a_count = road_a_data.get("vehicle_count", 0)
            road_a_density = road_a_data.get("density", "UNKNOWN")
            road_a_color = MOG2Analyzer.get_density_color(road_a_density)
            
            cv2.putText(result, "ROAD A", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(result, f"Vehicles: {road_a_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, road_a_color, 2)
            cv2.putText(result, f"Density: {road_a_density}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, road_a_color, 2)
            
            # Road B statistics (right side)
            road_b_data = road_stats.get("Road B", {})
            road_b_count = road_b_data.get("vehicle_count", 0)
            road_b_density = road_b_data.get("density", "UNKNOWN")
            road_b_color = MOG2Analyzer.get_density_color(road_b_density)
            
            right_x = frame.shape[1] - 250
            cv2.putText(result, "ROAD B", (right_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(result, f"Vehicles: {road_b_count}", (right_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, road_b_color, 2)
            cv2.putText(result, f"Density: {road_b_density}", (right_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, road_b_color, 2)
            
            # Total
            total = road_stats.get("total_vehicles", 0)
            cv2.putText(result, f"Total: {total}", (frame.shape[1]//2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Fallback statistics without road_stats
            cv2.putText(
                result, f"MOG2: {len(mog2_boxes)} | YOLO: {len(yolo_boxes) if yolo_boxes else 0}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
        
        return result
