"""Tests for Member 3 SQLite event persistence and review status updates."""

import json
from pathlib import Path
import tempfile
import unittest

from src.database.database import (
    event_identity_key,
    get_event_counts,
    get_events,
    get_pending_events,
    initialize_database,
    insert_event,
    update_event_status,
)


class TestEventDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_event(self, **overrides):
        event = {
            "event_type": "sustained_proximity",
            "track_ids": [5, 2],
            "timestamp": 12.7,
            "frame_id": 381,
            "camera_id": "CAM_TEST",
            "confidence": 0.84,
            "status": "pending",
        }
        event.update(overrides)
        return event

    def test_initialization_creates_events_table(self):
        initialize_database(self.db_path)
        self.assertEqual(get_events(db_path=self.db_path), [])

    def test_insert_retrieves_event_and_json_track_ids(self):
        event_id = insert_event(self.make_event(), db_path=self.db_path)
        events = get_events(db_path=self.db_path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["id"], event_id)
        self.assertEqual(events[0]["track_ids"], [5, 2])
        self.assertEqual(events[0]["video_timestamp"], 12.7)
        self.assertEqual(events[0]["status"], "pending")
        self.assertEqual(json.loads(json.dumps(events[0]["track_ids"])), [5, 2])

    def test_pending_event_retrieval(self):
        insert_event(self.make_event(), db_path=self.db_path)
        insert_event(
            self.make_event(
                event_type="potential_distress_gesture",
                frame_id=382,
                timestamp=12.8,
            ),
            db_path=self.db_path,
        )
        self.assertEqual(len(get_pending_events(db_path=self.db_path)), 2)

    def test_verify_and_dismiss_workflow(self):
        verified_id = insert_event(self.make_event(), db_path=self.db_path)
        dismissed_id = insert_event(
            self.make_event(frame_id=383, timestamp=12.9),
            db_path=self.db_path,
        )

        self.assertTrue(update_event_status(verified_id, "verified", self.db_path))
        self.assertTrue(update_event_status(dismissed_id, "dismissed", self.db_path))
        self.assertEqual(
            {event["id"]: event["status"] for event in get_events(db_path=self.db_path)},
            {verified_id: "verified", dismissed_id: "dismissed"},
        )
        self.assertEqual(get_pending_events(db_path=self.db_path), [])

    def test_invalid_status_is_rejected(self):
        event_id = insert_event(self.make_event(), db_path=self.db_path)
        with self.assertRaises(ValueError):
            update_event_status(event_id, "confirmed", self.db_path)
        with self.assertRaises(ValueError):
            insert_event(self.make_event(status="confirmed"), db_path=self.db_path)

    def test_event_counts(self):
        pending_id = insert_event(self.make_event(), db_path=self.db_path)
        verified_id = insert_event(
            self.make_event(frame_id=382, timestamp=12.8),
            db_path=self.db_path,
        )
        dismissed_id = insert_event(
            self.make_event(frame_id=383, timestamp=12.9),
            db_path=self.db_path,
        )
        update_event_status(verified_id, "verified", self.db_path)
        update_event_status(dismissed_id, "dismissed", self.db_path)

        self.assertEqual(
            get_event_counts(self.db_path),
            {"pending": 1, "verified": 1, "dismissed": 1, "total": 3},
        )
        self.assertEqual(get_events(status="pending", db_path=self.db_path)[0]["id"], pending_id)

    def test_duplicate_events_are_not_inserted_twice(self):
        event = self.make_event()
        first_id = insert_event(event, db_path=self.db_path)
        duplicate_id = insert_event(event, db_path=self.db_path)

        self.assertEqual(first_id, duplicate_id)
        self.assertEqual(len(get_events(db_path=self.db_path)), 1)

    def test_identity_key_ignores_track_id_order(self):
        first = self.make_event(track_ids=[2, 5])
        second = self.make_event(track_ids=[5, 2])
        self.assertEqual(event_identity_key(first), event_identity_key(second))

    def test_optional_event_fields_are_supported(self):
        event_id = insert_event(
            {"event_type": "sustained_proximity"},
            db_path=self.db_path,
        )
        stored = get_events(db_path=self.db_path)[0]
        self.assertEqual(stored["id"], event_id)
        self.assertEqual(stored["track_ids"], [])
        self.assertIsNone(stored["frame_id"])
        self.assertIsNone(stored["confidence"])
        self.assertEqual(stored["camera_id"], "CAM_01")


if __name__ == "__main__":
    unittest.main()
