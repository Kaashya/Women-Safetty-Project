"""
Safety Event Manager module.

Responsible for standardizing, validating, and formatting potential safety events
emitted by Member 2 detectors for handoff to Member 3 (Dashboard / SQLite).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np


# Standard event type constants
EVENT_POTENTIAL_DISTRESS = "potential_distress_gesture"
EVENT_SUSTAINED_PROXIMITY = "sustained_proximity"


class SafetyEventManager:
    """
    Lightweight factory and manager for standardized safety event dictionaries.

    Ensures that every generated event satisfies the uniform contract expected
    by Member 3's triage dashboard and database layer.
    """

    def __init__(self, default_camera_id: str = "CAM_01") -> None:
        """
        Initialize the event manager.

        Args:
            default_camera_id: Identifier for the CCTV stream source.
        """
        self.default_camera_id = str(default_camera_id)

    def create_event(
        self,
        event_type: str,
        track_ids: Union[int, List[int], None],
        timestamp: float,
        frame_id: int,
        camera_id: Optional[str] = None,
        confidence: float = 0.8,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized safety event dictionary.

        Args:
            event_type: Identifier of the event ("potential_distress_gesture", "sustained_proximity").
            track_ids: Tracking ID or list of IDs involved in the event.
            timestamp: Video timestamp in seconds when the event was detected.
            frame_id: Frame sequence number.
            camera_id: CCTV camera identifier (defaults to instance camera_id).
            confidence: Heuristic event score between 0.0 and 1.0.
            details: Optional dictionary containing diagnostic metrics (duration, distance, etc.).

        Returns:
            Standardized event dictionary:
            {
                "event_type": str,
                "track_ids": List[int],
                "timestamp": float,
                "frame_id": int,
                "camera_id": str,
                "confidence": float,
                "status": "pending",
                "details": dict
            }
        """
        # Normalize track_ids into a list of ints
        norm_track_ids: List[int] = []
        if isinstance(track_ids, (list, tuple, set)):
            for tid in track_ids:
                if tid is not None:
                    try:
                        norm_track_ids.append(int(tid))
                    except (ValueError, TypeError):
                        pass
        elif track_ids is not None:
            try:
                norm_track_ids.append(int(track_ids))
            except (ValueError, TypeError):
                pass

        # Clamp confidence strictly between 0.0 and 1.0
        clamped_conf = float(np.clip(float(confidence), 0.0, 1.0))

        cam_id = str(camera_id if camera_id is not None else self.default_camera_id)

        event = {
            "event_type": str(event_type),
            "track_ids": norm_track_ids,
            "timestamp": round(float(timestamp), 2),
            "frame_id": int(frame_id),
            "camera_id": cam_id,
            "confidence": round(clamped_conf, 2),
            "status": "pending",  # Standard initial triage status for Member 3
            "details": details if details is not None else {}
        }
        return event

    @staticmethod
    def format_event_label(event: Dict[str, Any]) -> str:
        """
        Format a human-readable display label for overlays or logs.

        Examples:
            - "POTENTIAL DISTRESS — Person 2 (Conf: 0.85)"
            - "SUSTAINED PROXIMITY — Persons 3 & 5 (Conf: 0.78)"
        """
        etype = event.get("event_type", "safety_event")
        conf = float(event.get("confidence", 0.0))
        tids = event.get("track_ids", [])

        if etype == EVENT_POTENTIAL_DISTRESS:
            target = f"Person {tids[0]}" if tids else "Person"
            return f"POTENTIAL DISTRESS — {target} (Conf: {conf:.2f})"
        elif etype == EVENT_SUSTAINED_PROXIMITY:
            if len(tids) >= 2:
                target = f"Persons {tids[0]} & {tids[1]}"
            elif tids:
                target = f"Person {tids[0]}"
            else:
                target = "Unknown Persons"
            return f"SUSTAINED PROXIMITY — {target} (Conf: {conf:.2f})"
        else:
            return f"{etype.upper()} (Conf: {conf:.2f})"
