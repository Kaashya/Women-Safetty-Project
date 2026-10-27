"""
MediaPipe Pose estimation module for Member 2.

Key Principles:
    - The MediaPipe Pose model is initialized ONCE and reused across all crops/frames.
    - Operates on cropped person regions extracted via `get_person_crop`.
    - Handles invalid, empty, or degenerate crops safely without crashing.
    - Exposes a clean, structured landmark representation for downstream safety analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass
class PoseLandmarkPoint:
    """Represents a single 3D pose landmark with visibility score."""
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    """Structured pose landmarks extracted from a person crop."""
    left_wrist: PoseLandmarkPoint
    right_wrist: PoseLandmarkPoint
    left_shoulder: PoseLandmarkPoint
    right_shoulder: PoseLandmarkPoint
    left_elbow: Optional[PoseLandmarkPoint] = None
    right_elbow: Optional[PoseLandmarkPoint] = None
    nose: Optional[PoseLandmarkPoint] = None
    left_hip: Optional[PoseLandmarkPoint] = None
    right_hip: Optional[PoseLandmarkPoint] = None
    raw_landmarks: Optional[Any] = None


class PoseDetector:
    """
    Lightweight MediaPipe Pose wrapper for person pose landmark extraction.

    Reuses a single persistent Pose model instance across frames and bounding box crops.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0
    ) -> None:
        """
        Initialize the MediaPipe Pose model ONCE.

        Args:
            min_detection_confidence: Confidence threshold for person pose detection.
            min_tracking_confidence: Confidence threshold for landmark tracking.
            model_complexity: MediaPipe model complexity: 0 (fastest / Lite), 1 (Medium), 2 (Heavy).
        """
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.model_complexity = int(model_complexity)

        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            # static_image_mode=True is ideal when processing independent crops of multiple people
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        except Exception as e:
            self.pose_model = None
            self.mp_pose = None
            print(f"[WARNING] MediaPipe Pose initialization failed: {e}")

    def extract_pose(self, crop: Optional[np.ndarray]) -> Optional[PoseResult]:
        """
        Extract pose landmarks from a cropped person image.

        Args:
            crop: BGR image crop of a detected person.

        Returns:
            PoseResult if landmarks are successfully detected, else None.
        """
        if self.pose_model is None:
            return None

        # Validate input crop
        if (
            crop is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
            or crop.ndim < 3
        ):
            return None

        h, w = crop.shape[:2]
        # Skip crops that are too small for reliable pose estimation
        if h < 10 or w < 10:
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.pose_model.process(rgb_crop)
        except Exception:
            # Defensive catch to never crash on corrupted image buffers
            return None

        if not results or not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        mp_pl = self.mp_pose.PoseLandmark

        def _get_pt(landmark_id: Any) -> PoseLandmarkPoint:
            lm = landmarks[landmark_id.value]
            return PoseLandmarkPoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(getattr(lm, "visibility", 1.0))
            )

        try:
            return PoseResult(
                left_wrist=_get_pt(mp_pl.LEFT_WRIST),
                right_wrist=_get_pt(mp_pl.RIGHT_WRIST),
                left_shoulder=_get_pt(mp_pl.LEFT_SHOULDER),
                right_shoulder=_get_pt(mp_pl.RIGHT_SHOULDER),
                left_elbow=_get_pt(mp_pl.LEFT_ELBOW),
                right_elbow=_get_pt(mp_pl.RIGHT_ELBOW),
                nose=_get_pt(mp_pl.NOSE),
                left_hip=_get_pt(mp_pl.LEFT_HIP),
                right_hip=_get_pt(mp_pl.RIGHT_HIP),
                raw_landmarks=results.pose_landmarks
            )
        except Exception:
            return None

    def close(self) -> None:
        """Release underlying MediaPipe resources cleanly."""
        if self.pose_model is not None:
            try:
                self.pose_model.close()
            except Exception:
                pass
            self.pose_model = None
