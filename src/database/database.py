"""SQLite persistence for potential safety events and human review status."""

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Union


DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "events.db"
EVENT_STATUSES = {"pending", "verified", "dismissed"}
DEFAULT_CAMERA_ID = "CAM_01"
DatabasePath = Union[str, Path]


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _normalize_track_ids(track_ids: Any) -> List[int]:
    if track_ids is None:
        return []
    values = track_ids if isinstance(track_ids, (list, tuple, set)) else [track_ids]
    normalized = []
    for value in values:
        if value is None:
            continue
        try:
            normalized.append(int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return normalized


def _event_timestamp(event: Mapping[str, Any]) -> float:
    try:
        timestamp = float(event.get("timestamp", event.get("video_timestamp", 0.0)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Event timestamp must be a finite number.") from error
    if not math.isfinite(timestamp):
        raise ValueError("Event timestamp must be a finite number.")
    return timestamp


def event_identity_key(event: Mapping[str, Any]) -> str:
    """Return a stable key so Streamlit reruns cannot persist the same event twice."""
    event_type = str(event.get("event_type", "")).strip()
    if not event_type:
        raise ValueError("Event type is required.")

    frame_id = event.get("frame_id")
    if frame_id is not None:
        try:
            frame_id = int(frame_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Frame ID must be an integer or None.") from error

    identity = {
        "event_type": event_type,
        "track_ids": sorted(_normalize_track_ids(event.get("track_ids"))),
        "timestamp": _event_timestamp(event),
        "frame_id": frame_id,
        "camera_id": str(event.get("camera_id") or DEFAULT_CAMERA_ID),
        "source_video": str(event.get("source_video") or ""),
    }
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()


def initialize_database(db_path: DatabasePath = DATABASE_PATH) -> None:
    """Create the event table and its unique rerun-deduplication index."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    track_ids TEXT NOT NULL,
                    video_timestamp REAL NOT NULL,
                    frame_id INTEGER,
                    camera_id TEXT NOT NULL,
                    confidence REAL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'verified', 'dismissed')),
                    identity_key TEXT NOT NULL UNIQUE,
                    source_video TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def insert_event(
    event: Mapping[str, Any],
    db_path: DatabasePath = DATABASE_PATH,
) -> int:
    """Persist an event, returning its row ID (existing ID for a duplicate)."""
    if not isinstance(event, Mapping):
        raise TypeError("Event must be a mapping.")

    event_type = str(event.get("event_type", "")).strip()
    if not event_type:
        raise ValueError("Event type is required.")

    timestamp = _event_timestamp(event)
    frame_id = event.get("frame_id")
    if frame_id is not None:
        try:
            frame_id = int(frame_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Frame ID must be an integer or None.") from error

    confidence = event.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Event score must be a finite number or None.") from error
        if not math.isfinite(confidence):
            raise ValueError("Event score must be a finite number or None.")

    camera_id = str(event.get("camera_id") or DEFAULT_CAMERA_ID)
    track_ids = _normalize_track_ids(event.get("track_ids"))
    source_video = event.get("source_video")
    if source_video is not None:
        source_video = str(source_video)

    status = str(event.get("status", "pending"))
    if status not in EVENT_STATUSES:
        raise ValueError(
            "Event status must be one of: pending, verified, dismissed."
        )

    normalized_event = {
        "event_type": event_type,
        "track_ids": track_ids,
        "timestamp": timestamp,
        "frame_id": frame_id,
        "camera_id": camera_id,
        "source_video": source_video,
    }
    identity_key = event_identity_key(normalized_event)
    track_ids_json = json.dumps(track_ids)

    with closing(_connect(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    event_type, track_ids, video_timestamp, frame_id, camera_id,
                    confidence, status, identity_key, source_video
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity_key) DO NOTHING
                """,
                (
                    event_type,
                    track_ids_json,
                    timestamp,
                    frame_id,
                    camera_id,
                    confidence,
                    status,
                    identity_key,
                    source_video,
                ),
            )
            if cursor.rowcount == 1:
                return int(cursor.lastrowid)
            row = connection.execute(
                "SELECT id FROM events WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("Event could not be inserted or retrieved.")
            return int(row["id"])


def _event_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    event = dict(row)
    event["track_ids"] = json.loads(event["track_ids"])
    return event


def get_events(
    status: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: DatabasePath = DATABASE_PATH,
) -> List[Dict[str, Any]]:
    """Retrieve newest events first, optionally filtered by status or type."""
    if status is not None and status not in EVENT_STATUSES:
        raise ValueError("Unknown event status.")
    if limit is not None and limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    conditions = []
    parameters = []
    if status is not None:
        conditions.append("status = ?")
        parameters.append(status)
    if event_type is not None:
        conditions.append("event_type = ?")
        parameters.append(event_type)

    query = "SELECT * FROM events"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)

    with closing(_connect(db_path)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_event_from_row(row) for row in rows]


def get_pending_events(
    db_path: DatabasePath = DATABASE_PATH,
) -> List[Dict[str, Any]]:
    """Retrieve all events awaiting a human decision."""
    return get_events(status="pending", db_path=db_path)


def update_event_status(
    event_id: int,
    status: str,
    db_path: DatabasePath = DATABASE_PATH,
) -> bool:
    """Move a pending event to a valid review status; return whether it changed."""
    if status not in EVENT_STATUSES:
        raise ValueError("Status must be pending, verified, or dismissed.")

    with closing(_connect(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE events
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'pending'
                """,
                (status, event_id),
            )
            return cursor.rowcount == 1


def get_event_counts(db_path: DatabasePath = DATABASE_PATH) -> Dict[str, int]:
    """Return counts for all review statuses and total events."""
    counts = {"pending": 0, "verified": 0, "dismissed": 0, "total": 0}
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM events GROUP BY status"
        ).fetchall()
    for row in rows:
        counts[row["status"]] = int(row["count"])
    counts["total"] = sum(counts[status] for status in EVENT_STATUSES)
    return counts
