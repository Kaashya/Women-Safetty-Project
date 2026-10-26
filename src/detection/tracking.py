"""
Tracking utilities and result parsers for multi-person tracking.
"""

from typing import Any, List, Optional
import numpy as np

from src.utils.helpers import clamp_bbox


def format_tracking_label(track_id: Optional[int], confidence: float) -> str:
    """
    Format display label for a detected person.

    Examples:
        - track_id=1, conf=0.91 -> "Person 1 — 0.91"
        - track_id=None, conf=0.87 -> "Person — 0.87"

    Args:
        track_id: Tracking ID integer, or None if tracking is unavailable/unassigned.
        confidence: Detection confidence score between 0.0 and 1.0.

    Returns:
        Formatted label string.
    """
    if track_id is not None:
        return f"Person {track_id} — {confidence:.2f}"
    return f"Person — {confidence:.2f}"


def extract_person_detections(
    boxes: Any,
    frame_width: int,
    frame_height: int
) -> List[dict]:
    """
    Extract structured person detections from Ultralytics detection/tracking boxes.

    Safely extracts:
        - bounding box coordinates [x1, y1, x2, y2]
        - tracking ID (int or None)
        - confidence score (float)

    Handles missing tracking IDs, empty detections, and out-of-bound coordinates gracefully.

    Args:
        boxes: Ultralytics Boxes object (results[0].boxes).
        frame_width: Image width in pixels.
        frame_height: Image height in pixels.

    Returns:
        List of dictionaries conforming to the detection contract:
        [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.91
            },
            ...
        ]
    """
    if boxes is None or len(boxes) == 0:
        return []

    people: List[dict] = []

    # Extract coordinates as numpy array
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
    except AttributeError:
        # If passed raw numpy or list for mocking/testing
        xyxy_arr = getattr(boxes, "xyxy", [])
        conf_arr = getattr(boxes, "conf", [])

    # Extract tracking IDs if present
    track_ids = None
    if hasattr(boxes, "id") and boxes.id is not None:
        try:
            track_ids = boxes.id.int().cpu().numpy()
        except (AttributeError, TypeError):
            try:
                track_ids = np.asarray(boxes.id, dtype=int)
            except Exception:
                track_ids = None

    for i in range(len(xyxy_arr)):
        raw_box = xyxy_arr[i]
        clamped = clamp_bbox(raw_box, width=frame_width, height=frame_height)
        if clamped is None:
            continue

        conf = float(conf_arr[i]) if i < len(conf_arr) else 0.0

        track_id = None
        if track_ids is not None and i < len(track_ids):
            try:
                tid = int(track_ids[i])
                # Filter out negative or placeholder IDs if any
                if tid >= 0:
                    track_id = tid
            except (ValueError, TypeError):
                track_id = None

        people.append({
            "track_id": track_id,
            "bbox": clamped,
            "confidence": round(conf, 4)
        })

    return people
