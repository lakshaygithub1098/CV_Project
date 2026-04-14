"""Detection module - Vehicle detection using YOLO and MOG2"""
from .yolo_detector import YOLODetector
from .mog2_detector import MOG2OnlyDetector as MOG2Detector
from .hybrid_detector import HybridDetector

__all__ = ["YOLODetector", "MOG2Detector", "HybridDetector"]
