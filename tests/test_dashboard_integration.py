"""Tests for dashboard video setup and Member 1/2 frame handoff."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.dashboard import app as dashboard_app


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class TestDashboardIntegration(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.session_state_patch = patch.object(
            dashboard_app.st,
            "session_state",
            self.state,
        )
        self.session_state_patch.start()
        dashboard_app._initialize_session_state()

    def tearDown(self):
        if self.state.get("_temporary_video_path"):
            dashboard_app._reset_analysis()
        self.session_state_patch.stop()

    def test_video_start_initializes_pipeline_once_and_reset_cleans_file(self):
        state = self.state
        uploaded_file = MagicMock()
        uploaded_file.name = "sample.mp4"
        uploaded_file.getvalue.return_value = b"synthetic-video-data"

        fake_processor = MagicMock()
        fake_safety_detector = MagicMock()
        fake_safety_detector.pose_detector.pose_model = object()

        with patch.object(
            dashboard_app,
            "VideoProcessor",
            return_value=fake_processor,
        ) as video_cls:
            with patch.object(dashboard_app, "PersonDetector") as person_cls:
                with patch.object(
                    dashboard_app,
                    "SafetyEventDetector",
                    return_value=fake_safety_detector,
                ) as safety_cls:
                    error = dashboard_app._start_analysis(uploaded_file, "CAM_TEST")

        self.assertIsNone(error)
        self.assertTrue(state.is_processing)
        self.assertEqual(state.active_camera_id, "CAM_TEST")
        self.assertEqual(state.source_video, "sample.mp4")
        self.assertTrue(Path(state["_temporary_video_path"]).is_file())
        video_cls.assert_called_once()
        person_cls.assert_called_once_with(confidence=0.4, track=True)
        safety_cls.assert_called_once_with(camera_id="CAM_TEST")

        temp_path = Path(state["_temporary_video_path"])
        dashboard_app._reset_analysis()
        fake_processor.release.assert_called_once()
        fake_safety_detector.close.assert_called_once()
        self.assertFalse(temp_path.exists())
        self.assertFalse(state.is_processing)
        self.assertIsNone(state["_person_detector"])

    def test_frame_uses_existing_detectors_and_persists_member2_event(self):
        state = self.state
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame_data = {
            "frame_id": 7,
            "timestamp": 0.24,
            "frame": frame,
            "annotated_frame": frame.copy(),
            "people": [{"track_id": 4, "bbox": [1, 1, 8, 10], "confidence": 0.9}],
        }
        event = {
            "event_type": "sustained_proximity",
            "track_ids": [4, 9],
            "timestamp": 0.24,
            "frame_id": 7,
            "confidence": 0.75,
            "status": "pending",
        }
        processor = MagicMock()
        processor.read_frame.return_value = (True, frame, 7, 0.24)
        person_detector = MagicMock()
        person_detector.detect_frame.return_value = frame_data
        safety_detector = MagicMock()
        safety_detector.process_frame.return_value = [event]
        safety_detector.annotate_events.return_value = frame.copy()

        state["_video_processor"] = processor
        state["_person_detector"] = person_detector
        state["_safety_detector"] = safety_detector
        state.active_camera_id = "CAM_TEST"
        state.source_video = "sample.mp4"

        with patch.object(dashboard_app, "insert_event") as persist:
            end_of_video = dashboard_app._process_next_frame()

        self.assertFalse(end_of_video)
        person_detector.detect_frame.assert_called_once_with(
            frame,
            frame_id=7,
            timestamp=0.24,
        )
        safety_detector.process_frame.assert_called_once_with(frame_data)
        safety_detector.annotate_events.assert_called_once_with(
            frame_data["annotated_frame"],
            events=[event],
            people=frame_data["people"],
        )
        persisted_event = persist.call_args.args[0]
        self.assertEqual(persisted_event["camera_id"], "CAM_TEST")
        self.assertEqual(persisted_event["source_video"], "sample.mp4")
        self.assertEqual(state.current_frame_id, 7)
        self.assertEqual(state.current_people_count, 1)
        self.assertIsNotNone(state.last_frame_rgb)


if __name__ == "__main__":
    unittest.main()
