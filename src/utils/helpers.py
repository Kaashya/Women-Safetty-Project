"""
Helper functions for bounding-box manipulation, person image cropping,
and frame visualization.
"""

from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np


def clamp_bbox(
    bbox: Any,
    width: int,
    height: int
) -> Optional[List[int]]:
    """
    Validate and clamp bounding box coordinates to image dimensions [0, width] and [0, height].

    Args:
        bbox: Sequence of 4 coordinates [x1, y1, x2, y2].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of 4 clamped integers [x1, y1, x2, y2] if valid, or None if degenerate.
    """
    if bbox is None:
        return None

    try:
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    # Check for NaN / Inf
    for v in (x1, y1, x2, y2):
        if np.isnan(v) or np.isinf(v):
            return None

    # Clamp to boundaries
    ix1 = max(0, min(width, int(round(x1))))
    iy1 = max(0, min(height, int(round(y1))))
    ix2 = max(0, min(width, int(round(x2))))
    iy2 = max(0, min(height, int(round(y2))))

    # Ensure box has positive area
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return [ix1, iy1, ix2, iy2]


def get_person_crop(
    frame: np.ndarray,
    bbox: Any
) -> np.ndarray:
    """
    Safely crop a detected person region from a video frame.

    This helper provides safe coordinate clamping, dimension checks, and
    defensive error handling. Used primarily by Member 2 to extract person crops
    for MediaPipe Pose estimation and fall detection.

    Args:
        frame: Video frame as a NumPy array (H x W x C).
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Cropped person image (NumPy array). Returns an empty array (0, 0, C)
        if the bounding box or frame is invalid.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    clamped = clamp_bbox(bbox, width=w, height=h)

    if clamped is None:
        channels = frame.shape[2] if frame.ndim > 2 else 0
        return np.empty((0, 0, channels), dtype=frame.dtype)

    x1, y1, x2, y2 = clamped
    # Return a contiguous copy to avoid memory leaks or unintended slice mutation
    return frame[y1:y2, x1:x2].copy()


def draw_person_detections(
    frame: np.ndarray,
    people: List[dict],
    box_color: Tuple[int, int, int] = (0, 200, 50),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes, tracking IDs, and confidence scores on a copy of the frame.

    Labels follow the standard format:
        - With tracking ID: "Person 1 — 0.91"
        - Without tracking ID: "Person — 0.91"

    Args:
        frame: Original video frame (BGR NumPy array).
        people: List of detected person dictionaries containing 'bbox', 'confidence', 'track_id'.
        box_color: BGR tuple for bounding box edges.
        text_color: BGR tuple for text label.
        thickness: Line thickness for bounding box.

    Returns:
        Annotated copy of the frame.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for person in people:
        bbox = person.get("bbox")
        clamped = clamp_bbox(bbox, width=w, height=h)
        if clamped is None:
            continue

        x1, y1, x2, y2 = clamped
        track_id = person.get("track_id")
        conf = float(person.get("confidence", 0.0))

        # Format label
        if track_id is not None:
            label = f"Person {track_id} — {conf:.2f}"
        else:
            label = f"Person — {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw filled label background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Place label above box if space allows; otherwise inside the top edge
        if y1 - label_h - baseline - 4 >= 0:
            bg_y1 = y1 - label_h - baseline - 4
            bg_y2 = y1
            text_y = y1 - baseline - 2
        else:
            bg_y1 = y1
            bg_y2 = y1 + label_h + baseline + 4
            text_y = y1 + label_h + 2

        bg_x1 = x1
        bg_x2 = min(w, x1 + label_w + 6)

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (bg_x1 + 3, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated
