"""Tests for dashboard formatting and incomplete-event handling."""

import unittest

from src.dashboard.helpers import (
    event_label,
    format_event_score,
    format_timestamp,
    format_track_ids,
    normalize_track_ids,
)


class TestDashboardHelpers(unittest.TestCase):
    def test_event_labels(self):
        self.assertEqual(
            event_label("potential_distress_gesture"),
            "Potential Distress Gesture",
        )
        self.assertEqual(event_label("unknown_event"), "Unknown Event")
        self.assertEqual(event_label(None), "Potential Safety Event")

    def test_timestamp_formatting_and_missing_values(self):
        self.assertEqual(format_timestamp(12.7), "00:12.7")
        self.assertEqual(format_timestamp(3661.25), "01:01:01.2")
        self.assertEqual(format_timestamp(None), "Unknown")
        self.assertEqual(format_timestamp(float("nan")), "Unknown")

    def test_heuristic_score_formatting_and_missing_values(self):
        self.assertEqual(format_event_score(0.84), "84%")
        self.assertEqual(format_event_score(1.2), "100%")
        self.assertEqual(format_event_score(-0.2), "0%")
        self.assertEqual(format_event_score(None), "N/A")

    def test_missing_and_malformed_tracking_ids(self):
        self.assertEqual(normalize_track_ids(None), [])
        self.assertEqual(normalize_track_ids([2, None, "5", "invalid"]), [2, 5])
        self.assertEqual(format_track_ids([]), "Not available")
        self.assertEqual(format_track_ids([3, 5]), "3, 5")


if __name__ == "__main__":
    unittest.main()
