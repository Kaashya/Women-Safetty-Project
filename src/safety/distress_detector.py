"""
Predefined Distress Gesture Detection Module.

Prototype Distress Gesture Definition:
    Both hands/wrists intentionally raised above the shoulders/head region
    and held continuously for a sustained duration.

IMPORTANT DISCLAIMER:
    This is a predefined prototype gesture used to demonstrate safety-event
    recognition. It does NOT imply that every person raising their hands is in
    distress, nor does it establish that a crime or emergency occurred.
    All detections must be treated as potential safety events requiring human review.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.safety.event_manager import EVENT_POTENTIAL_DISTRESS, SafetyEventManager
from src.safety.pose_detector import PoseLandmarkPoint, PoseResult


@dataclass
class PersonGestureState:
    """Temporal tracking state for an individual person's gesture."""
    track_id: Optional[int]
    first_detected_ts: Optional[float] = None
    last_detected_ts: Optional[float] = None
    consecutive_frames: int = 0
    last_event_ts: Optional[float] = None
    history_visibility: List[float] = field(default_factory=list)


class DistressGestureDetector:
    """
    Detects sustained predefined distress gestures across tracked people.

    Maintains independent temporal state per tracking ID, enforces landmark
    visibility thresholds, requires temporal persistence, and applies cooldown
    to prevent duplicate alerts.
    """

    def __init__(
        self,
        event_manager: Optional[SafetyEventManager] = None,
        min_duration_sec: float = 1.0,
        min_consecutive_frames: int = 10,
        min_landmark_visibility: float = 0.5,
        shoulder_elevation_offset: float = 0.0,
        cooldown_sec: float = 5.0,
        stale_timeout_sec: float = 4.0
    ) -> None:
        """
        Args:
            event_manager: Standard safety event factory instance.
            min_duration_sec: Minimum seconds gesture must be held (default: 1.0s).
            min_consecutive_frames: Minimum frame count requirement (default: 10 frames).
            min_landmark_visibility: Minimum MediaPipe landmark visibility (default: 0.5).
            shoulder_elevation_offset: Additional margin wrists must be above shoulders.
            cooldown_sec: Seconds to suppress duplicate alerts for the same person (default: 5.0s).
            stale_timeout_sec: Seconds of inactivity before clearing a person's state (default: 4.0s).
        """
        self.event_manager = event_manager or SafetyEventManager()
        self.min_duration_sec = float(min_duration_sec)
        self.min_consecutive_frames = int(min_consecutive_frames)
        self.min_landmark_visibility = float(min_landmark_visibility)
        self.shoulder_elevation_offset = float(shoulder_elevation_offset)
        self.cooldown_sec = float(cooldown_sec)
        self.stale_timeout_sec = float(stale_timeout_sec)

        # State per tracking ID: Dict[int, PersonGestureState]
        self.states: Dict[int, PersonGestureState] = {}

    def is_both_hands_raised(
        self,
        pose: Optional[PoseResult]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate whether both hands are raised above the shoulders with sufficient visibility.

        In MediaPipe coordinates, y ranges from 0.0 (top) to 1.0 (bottom),
        so a wrist is above a shoulder when wrist.y < shoulder.y.

        Args:
            pose: PoseResult containing key landmarks.

        Returns:
            Tuple of (is_raised: bool, diagnostics: dict).
        """
        diag = {
            "left_wrist_above": False,
            "right_wrist_above": False,
            "visibility_ok": False,
            "avg_visibility": 0.0,
            "wrist_elevation": 0.0
        }

        if pose is None:
            return False, diag

        lw, rw = pose.left_wrist, pose.right_wrist
        ls, rs = pose.left_shoulder, pose.right_shoulder

        if None in (lw, rw, ls, rs):
            return False, diag

        # 1. Check landmark visibility
        visibilities = [lw.visibility, rw.visibility, ls.visibility, rs.visibility]
        avg_vis = float(np.mean(visibilities))
        diag["avg_visibility"] = round(avg_vis, 3)

        min_vis = min(visibilities)
        if min_vis < self.min_landmark_visibility:
            return False, diag
        diag["visibility_ok"] = True

        # 2. Check wrist elevation relative to shoulders
        # y decreases upward in image space
        left_above = lw.y < (ls.y - self.shoulder_elevation_offset)
        right_above = rw.y < (rs.y - self.shoulder_elevation_offset)

        diag["left_wrist_above"] = bool(left_above)
        diag["right_wrist_above"] = bool(right_above)

        if not (left_above and right_above):
            # Both hands MUST be raised. Single raised hand is rejected.
            return False, diag

        # Calculate average elevation above shoulders
        left_elev = max(0.0, ls.y - lw.y)
        right_elev = max(0.0, rs.y - rw.y)
        diag["wrist_elevation"] = round(float((left_elev + right_elev) / 2.0), 3)

        return True, diag

    def update(
        self,
        people: List[Dict[str, Any]],
        poses: Dict[Optional[int], Optional[PoseResult]],
        timestamp: float,
        frame_id: int
    ) -> List[Dict[str, Any]]:
        """
        Update temporal gesture tracking and detect potential distress events.

        Args:
            people: List of detected person dicts from Member 1.
            poses: Mapping of track_id to PoseResult.
            timestamp: Video timestamp in seconds.
            frame_id: Frame sequence number.

        Returns:
            List of generated potential distress safety events for this frame.
        """
        events: List[Dict[str, Any]] = []
        observed_ids = set()

        for person in people:
            track_id = person.get("track_id")
            if track_id is None:
                # Without a persistent tracking ID, do not accumulate cross-frame state
                continue

            observed_ids.add(track_id)
            pose = poses.get(track_id)
            is_raised, diag = self.is_both_hands_raised(pose)

            # Retrieve or initialize state for this person
            if track_id not in self.states:
                self.states[track_id] = PersonGestureState(track_id=track_id)
            state = self.states[track_id]
            state.last_detected_ts = timestamp

            if is_raised:
                if state.first_detected_ts is None:
                    state.first_detected_ts = timestamp
                    state.consecutive_frames = 1
                else:
                    state.consecutive_frames += 1

                held_duration = timestamp - state.first_detected_ts

                # Check temporal persistence requirements
                duration_met = held_duration >= self.min_duration_sec
                frames_met = state.consecutive_frames >= self.min_consecutive_frames

                if duration_met and frames_met:
                    # Check cooldown
                    in_cooldown = (
                        state.last_event_ts is not None
                        and (timestamp - state.last_event_ts) < self.cooldown_sec
                    )

                    if not in_cooldown:
                        # Compute heuristic confidence score (bounded [0.1, 0.98])
                        avg_vis = diag.get("avg_visibility", 0.5)
                        elevation = diag.get("wrist_elevation", 0.1)
                        duration_score = min(1.0, held_duration / (self.min_duration_sec * 2.0))
                        heur_conf = float(np.clip(
                            0.4 * avg_vis + 0.3 * min(1.0, elevation * 3.0) + 0.3 * duration_score,
                            0.4,
                            0.98
                        ))

                        event = self.event_manager.create_event(
                            event_type=EVENT_POTENTIAL_DISTRESS,
                            track_ids=[track_id],
                            timestamp=timestamp,
                            frame_id=frame_id,
                            confidence=heur_conf,
                            details={
                                "gesture_name": "both_hands_raised",
                                "held_duration_sec": round(held_duration, 2),
                                "consecutive_frames": state.consecutive_frames,
                                "wrist_elevation": elevation,
                                "landmark_visibility": avg_vis,
                                "disclaimer": "Potential distress gesture requiring human verification."
                            }
                        )
                        events.append(event)
                        state.last_event_ts = timestamp
            else:
                # Reset active gesture timer when hands are lowered
                state.first_detected_ts = None
                state.consecutive_frames = 0

        # Prune stale tracking records to prevent unbounded memory growth
        self._prune_stale_states(timestamp)

        return events

    def _prune_stale_states(self, current_ts: float) -> None:
        """Remove state records for persons not seen within stale_timeout_sec."""
        stale_ids = [
            tid for tid, state in self.states.items()
            if state.last_detected_ts is not None
            and (current_ts - state.last_detected_ts) > self.stale_timeout_sec
        ]
        for tid in stale_ids:
            del self.states[tid]
