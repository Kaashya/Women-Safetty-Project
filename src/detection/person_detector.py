"""
YOLO-based person detection and tracking module.
"""

from pathlib import Path
from typing import Generator, List, Optional, Union
import numpy as np
from ultralytics import YOLO

from src.detection.tracking import extract_person_detections
from src.detection.video_processor import VideoProcessor
from src.utils.helpers import draw_person_detections


class PersonDetector:
    """
    Lightweight YOLO person detector with multi-person tracking support.

    Key Principles:
        - Model is loaded ONCE during initialization and reused across all frames.
        - Person class only (COCO class 0). Unrelated classes are discarded.
        - Provides tracking persistence across sequential video frames.
        - Produces a strictly standardized structured output contract for downstream modules.
        - Never invokes GUI displays (e.g. cv2.imshow) directly, ensuring headless
          compatibility with Streamlit, CLI, and automated tests.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        track: bool = True,
        tracker: str = "bytetrack.yaml",
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the PersonDetector and load the YOLO model once.

        Args:
            model_path: Path or identifier for Ultralytics YOLO model (default: 'yolov8n.pt').
            confidence: Confidence threshold for detections (default: 0.4).
            track: Whether to enable tracking (default: True).
            tracker: Tracker configuration to use ('bytetrack.yaml' or 'botsort.yaml').
            device: Target device ('cpu', 'cuda', '0', etc.). None selects automatically.
        """
        self.model_path = model_path
        self.confidence = float(confidence)
        self.track = bool(track)
        self.tracker = tracker
        self.device = device

        # Load YOLO model ONCE
        self.model = YOLO(self.model_path)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 1,
        timestamp: float = 0.0
    ) -> dict:
        """
        Run person detection (and optional tracking) on a single video frame.

        Args:
            frame: Input image/frame in BGR format (H x W x C).
            frame_id: Frame sequence number (default: 1).
            timestamp: Elapsed video timestamp in seconds (default: 0.0).

        Returns:
            Structured dictionary matching the standard detection contract:
            {
                "frame_id": int,
                "timestamp": float,
                "frame": np.ndarray (original frame),
                "annotated_frame": np.ndarray (annotated frame with bboxes and labels),
                "people": [
                    {
                        "track_id": int or None,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float
                    },
                    ...
                ]
            }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "frame_id": frame_id,
                "timestamp": round(timestamp, 3),
                "frame": frame,
                "annotated_frame": frame,
                "people": []
            }

        h, w = frame.shape[:2]

        # Run inference: track if enabled, otherwise predict
        # classes=[0] filters exclusively for 'person' class in COCO dataset
        if self.track:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                classes=[0],
                conf=self.confidence,
                device=self.device,
                verbose=False
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=[0],
                conf=self.confidence,
                device=self.device,
                verbose=False
            )

        boxes = results[0].boxes if (results and len(results) > 0) else None
        people = extract_person_detections(boxes, frame_width=w, frame_height=h)

        # Generate annotated frame
        annotated_frame = draw_person_detections(frame, people)

        return {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "frame": frame,
            "annotated_frame": annotated_frame,
            "people": people
        }

    def process_video(
        self,
        video_source: Union[str, int, Path]
    ) -> Generator[dict, None, None]:
        """
        Stream detection and tracking outputs frame-by-frame for an entire video.

        Args:
            video_source: Path to video file or camera index.

        Yields:
            Standard detection dictionaries for each frame in the video stream.
        """
        with VideoProcessor(video_source) as processor:
            for frame_id, timestamp, frame in processor.iter_frames():
                yield self.detect_frame(frame, frame_id=frame_id, timestamp=timestamp)
