"""
Unified Safety Event Detector facade (Member 2 Primary Public API).

Coordinates MediaPipe Pose estimation, Predefined Distress Gesture detection,
and Sustained Proximity / Following heuristics into a single high-level API
designed for seamless handoff to Member 3 (Streamlit / Database).
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.safety.distress_detector import DistressGestureDetector
from src.safety.event_manager import (
    EVENT_POTENTIAL_DISTRESS,
    EVENT_SUSTAINED_PROXIMITY,
    SafetyEventManager
)
from src.safety.pose_detector import PoseDetector, PoseResult
from src.safety.proximity_detector import ProximityDetector
from src.utils.helpers import clamp_bbox, get_person_crop


class SafetyEventDetector:
    """
    Unified Member 2 Safety Analytics facade.

    Usage Example:
        from src.safety import SafetyEventDetector

        safety_detector = SafetyEventDetector(camera_id="CAM_01")
        for frame_data in person_detector.process_video(video_path):
            events = safety_detector.process_frame(frame_data)
            annotated_frame = safety_detector.annotate_events(
                frame_data["annotated_frame"],
                events
            )
    """

    def __init__(
        self,
        camera_id: str = "CAM_01",
        distress_hold_sec: float = 1.0,
        distress_min_frames: int = 10,
        distress_min_visibility: float = 0.5,
        distress_cooldown_sec: float = 5.0,
        proximity_dist_thresh: float = 1.3,
        proximity_duration_sec: float = 2.0,
        proximity_min_frames: int = 15,
        proximity_min_similarity: float = 0.4,
        proximity_cooldown_sec: float = 8.0,
        stale_timeout_sec: float = 4.0,
        debug: bool = False
    ) -> None:
        """
        Initialize Member 2 safety pipeline and its sub-detectors.

        Args:
            camera_id: CCTV camera stream identifier (default: "CAM_01").
            distress_hold_sec: Minimum seconds distress gesture must be held.
            distress_min_frames: Minimum frame count for distress gesture.
            distress_min_visibility: Landmark visibility threshold.
            distress_cooldown_sec: Seconds to suppress duplicate distress alerts for a person.
            proximity_dist_thresh: Max normalized distance for proximity (units of bbox height).
            proximity_duration_sec: Minimum seconds sustained proximity must persist.
            proximity_min_frames: Minimum frame count for sustained proximity.
            proximity_min_similarity: Minimum cosine similarity for correlated movement.
            proximity_cooldown_sec: Seconds to suppress duplicate proximity alerts for a pair.
            stale_timeout_sec: Timeout to purge inactive tracking states.
            debug: Whether to print verbose diagnostic debug messages.
        """
        self.camera_id = str(camera_id)
        self.debug = bool(debug)

        # Standard event manager
        self.event_manager = SafetyEventManager(default_camera_id=self.camera_id)

        # 1. Pose detector (initialized ONCE and reused)
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )

        # 2. Distress gesture detector
        self.distress_detector = DistressGestureDetector(
            event_manager=self.event_manager,
            min_duration_sec=distress_hold_sec,
            min_consecutive_frames=distress_min_frames,
            min_landmark_visibility=distress_min_visibility,
            cooldown_sec=distress_cooldown_sec,
            stale_timeout_sec=stale_timeout_sec
        )

        # 3. Sustained proximity & following detector
        self.proximity_detector = ProximityDetector(
            event_manager=self.event_manager,
            proximity_threshold=proximity_dist_thresh,
            min_duration_sec=proximity_duration_sec,
            min_consecutive_frames=proximity_min_frames,
            min_movement_similarity=proximity_min_similarity,
            cooldown_sec=proximity_cooldown_sec,
            stale_timeout_sec=stale_timeout_sec
        )

        # Running history of active events (recent events within display window)
        self.recent_events: List[Dict[str, Any]] = []

    def process_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single frame from Member 1's detection pipeline.

        Args:
            frame_data: Structured output from PersonDetector.detect_frame():
                {
                    "frame_id": int,
                    "timestamp": float,
                    "frame": np.ndarray (original frame),
                    "annotated_frame": np.ndarray (Member 1 annotated frame),
                    "people": [
                        {"track_id": int, "bbox": [x1, y1, x2, y2], "confidence": float},
                        ...
                    ]
                }

        Returns:
            List of newly detected safety event dictionaries conforming to the standard contract.
        """
        if not frame_data:
            return []

        frame = frame_data.get("frame")
        frame_id = int(frame_data.get("frame_id", 1))
        timestamp = float(frame_data.get("timestamp", 0.0))
        people = frame_data.get("people", [])

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # 1. Extract pose landmarks for each person crop
        poses: Dict[Optional[int], Optional[PoseResult]] = {}
        for person in people:
            tid = person.get("track_id")
            bbox = person.get("bbox")
            if bbox is not None:
                crop = get_person_crop(frame, bbox)
                poses[tid] = self.pose_detector.extract_pose(crop)
            else:
                poses[tid] = None

        # 2. Update distress gesture analysis
        distress_events = self.distress_detector.update(
            people=people,
            poses=poses,
            timestamp=timestamp,
            frame_id=frame_id
        )

        # 3. Update sustained proximity / following analysis
        proximity_events = self.proximity_detector.update(
            people=people,
            timestamp=timestamp,
            frame_id=frame_id
        )

        # 4. Aggregate events
        new_events = distress_events + proximity_events

        if self.debug and new_events:
            for ev in new_events:
                print(f"[DEBUG SAFETY] New event: {self.event_manager.format_event_label(ev)}")

        # Keep a short buffer of active events for visual annotation persistence
        for ev in new_events:
            self.recent_events.append(ev)
        # Prune events older than 3 seconds from visual overlay buffer
        self.recent_events = [
            ev for ev in self.recent_events
            if (timestamp - ev.get("timestamp", 0.0)) <= 3.0
        ]

        return new_events

    def annotate_events(
        self,
        frame: np.ndarray,
        events: Optional[List[Dict[str, Any]]] = None,
        people: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Draw visual safety event notifications and banners onto a copy of the frame.

        Preserves Member 1's bounding boxes while adding distinctive safety alert
        overlays that Member 3 can render directly.

        Args:
            frame: Video frame (typically Member 1's annotated_frame).
            events: List of safety events to draw (uses recent_events if None or empty).
            people: Optional detected people list to highlight bounding boxes.

        Returns:
            Annotated copy of the frame with safety event alerts.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        active_to_draw = events if (events is not None and len(events) > 0) else self.recent_events
        if not active_to_draw:
            return annotated

        # Highlight involved person bounding boxes if people list is provided
        if people:
            people_by_id = {p.get("track_id"): p.get("bbox") for p in people if p.get("track_id") is not None}
            for ev in active_to_draw:
                etype = ev.get("event_type")
                tids = ev.get("track_ids", [])
                # Amber-red for distress, Amber-orange for proximity
                box_color = (0, 0, 255) if etype == EVENT_POTENTIAL_DISTRESS else (0, 140, 255)

                for tid in tids:
                    bbox = people_by_id.get(tid)
                    if bbox:
                        clamped = clamp_bbox(bbox, width=w, height=h)
                        if clamped:
                            x1, y1, x2, y2 = clamped
                            # Draw thicker warning box over person
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

        # Draw alert banners on upper region of the frame
        banner_y = 50
        for ev in active_to_draw:
            etype = ev.get("event_type")
            label = self.event_manager.format_event_label(ev)

            # Colors (BGR)
            if etype == EVENT_POTENTIAL_DISTRESS:
                bg_color = (20, 20, 220)  # Red warning
            else:
                bg_color = (0, 120, 240)  # Orange alert

            text_color = (255, 255, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (tw, th), base = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw banner background rectangle
            x1 = 15
            y1 = banner_y - th - 6
            x2 = min(w - 15, x1 + tw + 16)
            y2 = banner_y + base + 4

            cv2.rectangle(annotated, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(
                annotated,
                label,
                (x1 + 8, banner_y),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA
            )

            banner_y += th + base + 16
            if banner_y > h - 40:
                break

        return annotated

    def close(self) -> None:
        """Release underlying pose detector and helper resources cleanly."""
        if self.pose_detector is not None:
            self.pose_detector.close()
