"""
Safety Event Detection & Analytics Module (Member 2).

Provides:
    - SafetyEventDetector: High-level unified facade for Member 3 integration.
    - PoseDetector: MediaPipe Pose landmark extractor.
    - DistressGestureDetector: Temporal detector for predefined distress gestures.
    - ProximityDetector: Multi-person normalized proximity and following heuristic.
    - SafetyEventManager: Standard safety event creation and contract validation.
"""

from src.safety.distress_detector import DistressGestureDetector, PersonGestureState
from src.safety.event_manager import (
    EVENT_POTENTIAL_DISTRESS,
    EVENT_SUSTAINED_PROXIMITY,
    SafetyEventManager
)
from src.safety.pose_detector import PoseDetector, PoseLandmarkPoint, PoseResult
from src.safety.proximity_detector import PairProximityState, PersonMotionState, ProximityDetector
from src.safety.safety_event_detector import SafetyEventDetector

__all__ = [
    "SafetyEventDetector",
    "PoseDetector",
    "PoseResult",
    "PoseLandmarkPoint",
    "DistressGestureDetector",
    "PersonGestureState",
    "ProximityDetector",
    "PersonMotionState",
    "PairProximityState",
    "SafetyEventManager",
    "EVENT_POTENTIAL_DISTRESS",
    "EVENT_SUSTAINED_PROXIMITY"
]
