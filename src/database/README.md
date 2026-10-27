# Member 3: SQLite Event History

The database module uses Python's built-in `sqlite3` and persists generated Member 2 events for dashboard review. The default path is `data/events.db`; the database directory is created automatically and database files are excluded by `.gitignore`.

## Schema

```sql
CREATE TABLE events (
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
);
```

`track_ids` is JSON-encoded, so a single track or a pair can be stored without a separate relational table. `identity_key` is a deterministic hash of event type, track IDs, video timestamp, frame ID, camera ID, and source video. The unique constraint makes event insertion idempotent across Streamlit reruns.

## API

```python
from src.database import (
    initialize_database,
    insert_event,
    get_events,
    get_pending_events,
    update_event_status,
    get_event_counts,
)

initialize_database()
event_id = insert_event(event)
pending = get_pending_events()
history = get_events(status="verified", limit=100)
updated = update_event_status(event_id, "verified")
counts = get_event_counts()
```

`update_event_status` accepts `pending`, `verified`, or `dismissed`, and changes only an event that is still pending. Invalid statuses raise `ValueError`. All values in SQL statements are bound parameters.

The tests use temporary database paths; they do not modify the runtime database.
