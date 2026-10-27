"""Pure formatting and normalization helpers shared by the dashboard and tests."""

import math
from typing import Any, List, Optional


EVENT_LABELS = {
    "potential_distress_gesture": "Potential Distress Gesture",
    "sustained_proximity": "Sustained Proximity",
}


def event_label(event_type: Any) -> str:
    value = str(event_type or "").strip()
    if not value:
        return "Potential Safety Event"
    return EVENT_LABELS.get(value, value.replace("_", " ").title())


def format_timestamp(value: Any) -> str:
    """Format seconds as MM:SS.s (or HH:MM:SS.s) without failing on missing data."""
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return "Unknown"
    if not math.isfinite(seconds):
        return "Unknown"
    seconds = max(0.0, seconds)
    whole_seconds = int(seconds)
    fraction = seconds - whole_seconds
    minutes, second_part = divmod(whole_seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{second_part + fraction:04.1f}"
    return f"{minutes:02d}:{second_part + fraction:04.1f}"


def format_event_score(value: Any) -> str:
    """Display a heuristic score as a percentage, not a probability claim."""
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(score):
        return "N/A"
    return f"{round(min(1.0, max(0.0, score)) * 100):.0f}%"


def normalize_track_ids(value: Any) -> List[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    track_ids = []
    for item in values:
        if item is None:
            continue
        try:
            track_ids.append(int(item))
        except (TypeError, ValueError, OverflowError):
            continue
    return track_ids


def format_track_ids(value: Any) -> str:
    track_ids = normalize_track_ids(value)
    if not track_ids:
        return "Not available"
    return ", ".join(str(track_id) for track_id in track_ids)


def optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None
