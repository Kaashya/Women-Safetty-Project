"""
Unit and integration tests for Member 2 Safety Analytics Pipeline.

Tests:
    1. Pose detector handles invalid crops (empty, None, tiny).
    2. Missing pose does not crash.
    3. Low-visibility landmarks do not generate distress.
    4. One raised hand does not generate distress.
    5. Both hands raised for one frame does not generate distress.
    6. Sustained both-hands-raised sequence generates one event.
    7. Gesture cooldown prevents repeated alerts.
    8. Two people briefly crossing do not trigger proximity.
    9. Sustained close pair triggers proximity.
    10. Different pairs maintain independent histories.
    11. Missing track IDs do not crash.
    12. Empty people list does not crash.
    13. Event structure contains all required fields.
    14. Event status defaults to 'pending'.
    15. Event scores remain between 0.0 and 1.0.
    16. Stale track/pair state is cleaned after timeout.
    17. Annotation overlay renders correctly.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.safety.distress_detector import DistressGestureDetector
from src.safety.event_manager import (
    EVENT_POTENTIAL_DISTRESS,
    EVENT_SUSTAINED_PROXIMITY,
    SafetyEventManager
)
from src.safety.pose_detector import PoseDetector, PoseLandmarkPoint, PoseResult
from src.safety.proximity_detector import ProximityDetector
from src.safety.safety_event_detector import SafetyEventDetector


def create_mock_pose(
    lw_y: float = 0.2,
    rw_y: float = 0.2,
    ls_y: float = 0.5,
    rs_y: float = 0.5,
    visibility: float = 0.95
) -> PoseResult:
    """Helper to create synthetic pose landmarks."""
    return PoseResult(
        left_wrist=PoseLandmarkPoint(x=0.4, y=lw_y, z=0.0, visibility=visibility),
        right_wrist=PoseLandmarkPoint(x=0.6, y=rw_y, z=0.0, visibility=visibility),
        left_shoulder=PoseLandmarkPoint(x=0.4, y=ls_y, z=0.0, visibility=visibility),
        right_shoulder=PoseLandmarkPoint(x=0.6, y=rs_y, z=0.0, visibility=visibility),
        nose=PoseLandmarkPoint(x=0.5, y=0.3, z=0.0, visibility=visibility)
    )


class TestPoseDetector(unittest.TestCase):
    """Test MediaPipe Pose detector handling of edge cases and invalid crops."""

    def setUp(self):
        self.detector = PoseDetector(model_complexity=0)

    def tearDown(self):
        self.detector.close()

    def test_invalid_crops_return_none(self):
        # None crop
        self.assertIsNone(self.detector.extract_pose(None))
        # Empty array
        self.assertIsNone(self.detector.extract_pose(np.empty((0, 0, 3), dtype=np.uint8)))
        # Tiny/degenerate crops (under 10x10)
        self.assertIsNone(self.detector.extract_pose(np.zeros((5, 5, 3), dtype=np.uint8)))
        # 1D or 2D array
        self.assertIsNone(self.detector.extract_pose(np.zeros((20, 20), dtype=np.uint8)))


class TestDistressGestureDetection(unittest.TestCase):
    """Test predefined distress gesture detection logic, persistence, and cooldown."""

    def setUp(self):
        self.mgr = SafetyEventManager(default_camera_id="CAM_TEST")
        self.detector = DistressGestureDetector(
            event_manager=self.mgr,
            min_duration_sec=1.0,
            min_consecutive_frames=5,
            min_landmark_visibility=0.5,
            cooldown_sec=4.0
        )

    def test_missing_pose_does_not_crash(self):
        people = [{"track_id": 1, "bbox": [10, 10, 50, 100], "confidence": 0.9}]
        poses = {1: None}
        events = self.detector.update(people, poses, timestamp=1.0, frame_id=1)
        self.assertEqual(events, [])

    def test_low_visibility_landmarks_no_distress(self):
        # Hands raised (y=0.2 vs shoulder y=0.5), but low visibility (0.3 < 0.5)
        low_vis_pose = create_mock_pose(lw_y=0.2, rw_y=0.2, ls_y=0.5, rs_y=0.5, visibility=0.3)
        is_raised, diag = self.detector.is_both_hands_raised(low_vis_pose)
        self.assertFalse(is_raised)
        self.assertFalse(diag["visibility_ok"])

    def test_one_raised_hand_no_distress(self):
        # Left hand raised (0.2 < 0.5), right hand down (0.7 > 0.5)
        one_hand_pose = create_mock_pose(lw_y=0.2, rw_y=0.7, ls_y=0.5, rs_y=0.5, visibility=0.9)
        is_raised, diag = self.detector.is_both_hands_raised(one_hand_pose)
        self.assertFalse(is_raised)
        self.assertTrue(diag["left_wrist_above"])
        self.assertFalse(diag["right_wrist_above"])

    def test_both_hands_raised_single_frame_no_distress(self):
        pose = create_mock_pose(lw_y=0.2, rw_y=0.2, ls_y=0.5, rs_y=0.5, visibility=0.9)
        people = [{"track_id": 1, "bbox": [10, 10, 50, 100], "confidence": 0.9}]
        poses = {1: pose}
        # Single frame at t=0.0 (duration = 0.0 < min_duration 1.0)
        events = self.detector.update(people, poses, timestamp=0.0, frame_id=1)
        self.assertEqual(events, [])

    def test_sustained_both_hands_raised_generates_one_event(self):
        pose = create_mock_pose(lw_y=0.2, rw_y=0.2, ls_y=0.5, rs_y=0.5, visibility=0.9)
        people = [{"track_id": 2, "bbox": [10, 10, 50, 100], "confidence": 0.9}]
        poses = {2: pose}

        all_events = []
        # Simulate 12 frames over 1.2 seconds (0.1s step)
        for i in range(12):
            ts = round(i * 0.1, 2)
            evs = self.detector.update(people, poses, timestamp=ts, frame_id=i + 1)
            all_events.extend(evs)

        # Exactly ONE event generated when threshold is met
        self.assertEqual(len(all_events), 1)
        event = all_events[0]
        self.assertEqual(event["event_type"], EVENT_POTENTIAL_DISTRESS)
        self.assertEqual(event["track_ids"], [2])
        self.assertGreaterEqual(event["confidence"], 0.4)
        self.assertLessEqual(event["confidence"], 1.0)
        self.assertEqual(event["status"], "pending")

    def test_gesture_cooldown_prevents_repeated_alerts(self):
        pose = create_mock_pose(lw_y=0.2, rw_y=0.2, ls_y=0.5, rs_y=0.5, visibility=0.9)
        people = [{"track_id": 3, "bbox": [10, 10, 50, 100], "confidence": 0.9}]
        poses = {3: pose}

        events_generated = 0
        # Continuous raised hands for 3.0s (30 frames)
        for i in range(30):
            ts = round(i * 0.1, 2)
            evs = self.detector.update(people, poses, timestamp=ts, frame_id=i + 1)
            events_generated += len(evs)

        # Within the 4.0s cooldown, exactly ONE event should trigger
        self.assertEqual(events_generated, 1)

        # Advance past cooldown to 6.0s with hands lowered, then raised again at 7.0s
        lowered_pose = create_mock_pose(lw_y=0.7, rw_y=0.7, ls_y=0.5, rs_y=0.5)
        self.detector.update(people, {3: lowered_pose}, timestamp=5.0, frame_id=50)

        # Raise hands again at t=7.0s for 1.2s
        subsequent_events = 0
        for i in range(12):
            ts = round(7.0 + i * 0.1, 2)
            evs = self.detector.update(people, poses, timestamp=ts, frame_id=70 + i)
            subsequent_events += len(evs)

        # A new event should trigger after cooldown expired and gesture re-occurred
        self.assertEqual(subsequent_events, 1)


class TestProximityAndFollowingDetection(unittest.TestCase):
    """Test multi-person proximity, normalized distance, movement consistency, and pairs."""

    def setUp(self):
        self.mgr = SafetyEventManager(default_camera_id="CAM_TEST")
        self.detector = ProximityDetector(
            event_manager=self.mgr,
            proximity_threshold=1.5,
            min_duration_sec=2.0,
            min_consecutive_frames=10,
            min_movement_speed=5.0,
            min_movement_similarity=0.4,
            cooldown_sec=5.0,
            stale_timeout_sec=3.0
        )

    def test_brief_crossing_does_not_trigger_proximity(self):
        # Two people cross paths in opposite directions:
        # Person 1 moves left to right (dx > 0)
        # Person 2 moves right to left (dx < 0)
        events = []
        for i in range(15):
            ts = round(i * 0.1, 2)
            p1_x = 100 + i * 10
            p2_x = 250 - i * 10
            people = [
                {"track_id": 1, "bbox": [p1_x, 100, p1_x + 40, 200], "confidence": 0.9},
                {"track_id": 2, "bbox": [p2_x, 100, p2_x + 40, 200], "confidence": 0.9}
            ]
            evs = self.detector.update(people, timestamp=ts, frame_id=i + 1)
            events.extend(evs)

        # Opposing directions -> cosine similarity is negative -> NO alert
        self.assertEqual(len(events), 0)

    def test_sustained_close_pair_triggers_proximity(self):
        # Two people moving together in the same direction (dx=6 px/frame, dy=0)
        # separated by 40 pixels (average height = 100 -> norm_dist = 0.4 <= 1.5)
        events = []
        for i in range(25):
            ts = round(i * 0.1, 2)
            base_x = 50 + i * 6
            people = [
                {"track_id": 10, "bbox": [base_x, 100, base_x + 40, 200], "confidence": 0.9},
                {"track_id": 20, "bbox": [base_x + 45, 100, base_x + 85, 200], "confidence": 0.9}
            ]
            evs = self.detector.update(people, timestamp=ts, frame_id=i + 1)
            events.extend(evs)

        # Should generate exactly ONE sustained_proximity event
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_type"], EVENT_SUSTAINED_PROXIMITY)
        self.assertEqual(sorted(event["track_ids"]), [10, 20])
        self.assertGreaterEqual(event["confidence"], 0.3)
        self.assertLessEqual(event["confidence"], 1.0)
        self.assertEqual(event["status"], "pending")

    def test_different_pairs_maintain_independent_histories(self):
        # 3 people: (1, 2) are close and moving together; Person 3 is far away
        events = []
        for i in range(25):
            ts = round(i * 0.1, 2)
            base_x = 50 + i * 6
            people = [
                {"track_id": 1, "bbox": [base_x, 100, base_x + 40, 200], "confidence": 0.9},
                {"track_id": 2, "bbox": [base_x + 30, 100, base_x + 70, 200], "confidence": 0.9},
                {"track_id": 3, "bbox": [base_x + 500, 100, base_x + 540, 200], "confidence": 0.9}
            ]
            evs = self.detector.update(people, timestamp=ts, frame_id=i + 1)
            events.extend(evs)

        # Exactly ONE event, for pair (1, 2) only
        self.assertEqual(len(events), 1)
        self.assertEqual(sorted(events[0]["track_ids"]), [1, 2])

    def test_canonical_pair_ordering(self):
        key1 = self.detector.canonical_pair_key(5, 2)
        key2 = self.detector.canonical_pair_key(2, 5)
        self.assertEqual(key1, (2, 5))
        self.assertEqual(key2, (2, 5))


class TestSafetyEventContractAndIntegration(unittest.TestCase):
    """Test facade, missing IDs, empty lists, schema validation, and stale cleanup."""

    def setUp(self):
        self.detector = SafetyEventDetector(
            camera_id="CAM_TEST",
            distress_hold_sec=0.5,
            distress_min_frames=3,
            proximity_duration_sec=0.5,
            proximity_min_frames=3,
            stale_timeout_sec=2.0
        )

    def tearDown(self):
        self.detector.close()

    def test_empty_people_list_does_not_crash(self):
        frame_data = {
            "frame_id": 1,
            "timestamp": 0.0,
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "annotated_frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "people": []
        }
        events = self.detector.process_frame(frame_data)
        self.assertEqual(events, [])

    def test_missing_track_ids_do_not_crash(self):
        frame_data = {
            "frame_id": 1,
            "timestamp": 0.0,
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "annotated_frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "people": [
                {"track_id": None, "bbox": [10, 10, 50, 90], "confidence": 0.85},
                {"track_id": None, "bbox": [55, 10, 95, 90], "confidence": 0.88}
            ]
        }
        events = self.detector.process_frame(frame_data)
        self.assertEqual(events, [])

    def test_event_structure_and_status(self):
        mgr = SafetyEventManager(default_camera_id="CAM_TEST")
        ev = mgr.create_event(
            event_type=EVENT_POTENTIAL_DISTRESS,
            track_ids=[5],
            timestamp=14.25,
            frame_id=420,
            confidence=0.88,
            details={"notes": "test"}
        )
        # Validate schema contract required by Member 3
        required_keys = ["event_type", "track_ids", "timestamp", "frame_id", "camera_id", "confidence", "status"]
        for key in required_keys:
            self.assertIn(key, ev)

        self.assertEqual(ev["status"], "pending")
        self.assertEqual(ev["track_ids"], [5])
        self.assertEqual(ev["camera_id"], "CAM_TEST")
        self.assertAlmostEqual(ev["confidence"], 0.88)

    def test_event_scores_bounded_zero_one(self):
        mgr = SafetyEventManager()
        # Test out of bounds inputs
        ev_high = mgr.create_event("test", [1], 1.0, 1, confidence=1.85)
        ev_low = mgr.create_event("test", [1], 1.0, 1, confidence=-0.5)

        self.assertLessEqual(ev_high["confidence"], 1.0)
        self.assertGreaterEqual(ev_low["confidence"], 0.0)

    def test_stale_track_and_pair_state_cleaned(self):
        detector = ProximityDetector(stale_timeout_sec=2.0)
        people = [{"track_id": 1, "bbox": [10, 10, 40, 90], "confidence": 0.9}]
        detector.update(people, timestamp=1.0, frame_id=1)
        self.assertIn(1, detector.person_states)

        # Advance timestamp past stale timeout (1.0 + 3.0 = 4.0 > 2.0 timeout)
        detector.update([], timestamp=4.0, frame_id=40)
        self.assertNotIn(1, detector.person_states)

    def test_annotate_events_visual_overlay(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        events = [{
            "event_type": EVENT_POTENTIAL_DISTRESS,
            "track_ids": [1],
            "timestamp": 2.0,
            "frame_id": 20,
            "camera_id": "CAM_01",
            "confidence": 0.85,
            "status": "pending"
        }]
        annotated = self.detector.annotate_events(
            frame,
            events=events,
            people=[{"track_id": 1, "bbox": [20, 20, 80, 160], "confidence": 0.9}]
        )
        self.assertEqual(annotated.shape, frame.shape)
        # Frame was modified with visual overlay
        self.assertTrue(np.any(annotated > 0))


if __name__ == "__main__":
    unittest.main()
