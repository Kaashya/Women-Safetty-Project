"""
Person detection, tracking, and video processing modules.
"""

from src.detection.person_detector import PersonDetector
from src.detection.tracking import extract_person_detections, format_tracking_label
from src.detection.video_processor import VideoProcessor

__all__ = [
    "PersonDetector",
    "VideoProcessor",
    "extract_person_detections",
    "format_tracking_label",
]
