"""
Unit and integration tests for Member 1 Detection Pipeline.

Tests:
    1. Bounding box clamping and coordinate validation.
    2. Safe person cropping (get_person_crop) edge cases.
    3. Visual annotation and label formatting (with and without tracking IDs).
    4. VideoProcessor invalid path handling and synthetic video reading.
    5. Standard detection output contract validation.
    6. Zero-person frame handling.
    7. Missing tracking ID handling.
    8. End-of-video behavior.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import torch

from src.detection.person_detector import PersonDetector
from src.detection.tracking import extract_person_detections, format_tracking_label
from src.detection.video_processor import VideoProcessor
from src.utils.helpers import clamp_bbox, draw_person_detections, get_person_crop


class TestBoundingBoxAndCropping(unittest.TestCase):
    """Test coordinate validation, clamping, and safe person cropping."""

    def setUp(self):
        # Create a sample synthetic frame (100 height x 200 width x 3 channels)
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # Paint a white square in the center
        self.frame[30:70, 50:150] = 255

    def test_clamp_bbox_normal(self):
        bbox = [10, 20, 100, 80]
        clamped = clamp_bbox(bbox, width=200, height=100)
        self.assertEqual(clamped, [10, 20, 100, 80])

    def test_clamp_bbox_out_of_bounds(self):
        # Coordinates exceeding width/height or below zero
        bbox = [-20, -10, 250, 150]
        clamped = clamp_bbox(bbox, width=200, height=100)
        self.assertEqual(clamped, [0, 0, 200, 100])

    def test_clamp_bbox_degenerate(self):
        # Inverted coordinates: x2 <= x1
        self.assertIsNone(clamp_bbox([100, 20, 50, 80], width=200, height=100))
        # Inverted coordinates: y2 <= y1
        self.assertIsNone(clamp_bbox([10, 80, 50, 20], width=200, height=100))
        # Zero area
        self.assertIsNone(clamp_bbox([50, 50, 50, 50], width=200, height=100))

    def test_clamp_bbox_invalid_inputs(self):
        self.assertIsNone(clamp_bbox(None, 200, 100))
        self.assertIsNone(clamp_bbox([10, 20], 200, 100))  # too short
        self.assertIsNone(clamp_bbox(["a", "b", "c", "d"], 200, 100))  # non-numeric
        self.assertIsNone(clamp_bbox([float("nan"), 0, 100, 100], 200, 100))

    def test_get_person_crop_valid(self):
        crop = get_person_crop(self.frame, [50, 30, 150, 70])
        self.assertEqual(crop.shape, (40, 100, 3))
        # Pixels inside painted area should be 255
        self.assertTrue(np.all(crop == 255))

    def test_get_person_crop_clamped_boundary(self):
        # Box extending beyond frame boundaries should be clamped safely
        crop = get_person_crop(self.frame, [-10, 0, 50, 50])
        self.assertEqual(crop.shape, (50, 50, 3))

    def test_get_person_crop_invalid_box(self):
        # Inverted or invalid boxes should return an empty array without raising
        crop = get_person_crop(self.frame, [100, 100, 50, 50])
        self.assertEqual(crop.shape, (0, 0, 3))

        crop_none = get_person_crop(self.frame, None)
        self.assertEqual(crop_none.shape, (0, 0, 3))

    def test_get_person_crop_invalid_frame(self):
        crop = get_person_crop(None, [10, 10, 50, 50])
        self.assertEqual(crop.size, 0)


class TestAnnotationAndFormatting(unittest.TestCase):
    """Test visual annotation and label formatting."""

    def test_format_tracking_label(self):
        # With ID
        self.assertEqual(format_tracking_label(1, 0.9123), "Person 1 — 0.91")
        # Without ID
        self.assertEqual(format_tracking_label(None, 0.8765), "Person — 0.88")

    def test_draw_person_detections(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        people = [
            {"track_id": 1, "bbox": [10, 10, 50, 50], "confidence": 0.95},
            {"track_id": None, "bbox": [60, 60, 90, 90], "confidence": 0.80}
        ]
        annotated = draw_person_detections(frame, people)
        self.assertEqual(annotated.shape, frame.shape)
        # Annotated frame should have been modified (drawn upon)
        self.assertTrue(np.any(annotated > 0))

    def test_draw_zero_persons(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated = draw_person_detections(frame, [])
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, annotated))


class TestVideoProcessor(unittest.TestCase):
    """Test video processor initialization, invalid path handling, and playback."""

    def test_invalid_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            VideoProcessor("data/nonexistent_video_path.mp4")

    def test_synthetic_video_reading_and_lifecycle(self):
        # Create a small synthetic video file in a temporary folder
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = Path(temp_dir) / "test_synth.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width, height, fps, num_frames = 64, 64, 10.0, 5

            writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
            for i in range(num_frames):
                dummy_frame = np.full((height, width, 3), i * 40, dtype=np.uint8)
                writer.write(dummy_frame)
            writer.release()

            # Test VideoProcessor with context manager
            with VideoProcessor(temp_video_path) as proc:
                self.assertEqual(proc.width, width)
                self.assertEqual(proc.height, height)
                self.assertAlmostEqual(proc.fps, fps, places=1)

                frames_read = 0
                for frame_id, timestamp, frame in proc.iter_frames():
                    frames_read += 1
                    self.assertEqual(frame_id, frames_read)
                    self.assertGreaterEqual(timestamp, 0.0)
                    self.assertEqual(frame.shape, (height, width, 3))

                self.assertEqual(frames_read, num_frames)

                # Reading past EOF should return success=False
                ret, frame, fid, ts = proc.read_frame()
                self.assertFalse(ret)
                self.assertIsNone(frame)


class TestTrackingExtraction(unittest.TestCase):
    """Test extraction of structured person data from Ultralytics boxes."""

    def test_extraction_with_tracking_ids(self):
        class MockBoxes:
            def __init__(self):
                self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]])
                self.conf = torch.tensor([0.92, 0.88])
                self.id = torch.tensor([1, 2])
            def __len__(self):
                return 2

        mock_boxes = MockBoxes()
        people = extract_person_detections(mock_boxes, frame_width=100, frame_height=100)
        self.assertEqual(len(people), 2)
        self.assertEqual(people[0]["track_id"], 1)
        self.assertEqual(people[0]["bbox"], [10, 10, 50, 50])
        self.assertEqual(people[0]["confidence"], 0.92)
        self.assertEqual(people[1]["track_id"], 2)

    def test_extraction_missing_tracking_ids(self):
        class MockBoxesNoId:
            def __init__(self):
                self.xyxy = torch.tensor([[15.0, 20.0, 45.0, 80.0]])
                self.conf = torch.tensor([0.75])
                self.id = None
            def __len__(self):
                return 1

        mock_boxes = MockBoxesNoId()
        people = extract_person_detections(mock_boxes, frame_width=100, frame_height=100)
        self.assertEqual(len(people), 1)
        self.assertIsNone(people[0]["track_id"])
        self.assertEqual(people[0]["bbox"], [15, 20, 45, 80])
        self.assertEqual(people[0]["confidence"], 0.75)

    def test_extraction_empty_boxes(self):
        people = extract_person_detections(None, frame_width=100, frame_height=100)
        self.assertEqual(people, [])


class TestPersonDetectorContract(unittest.TestCase):
    """Test PersonDetector contract, output schema, and single model instantiation."""

    @patch("src.detection.person_detector.YOLO")
    def test_detection_contract_and_schema(self, mock_yolo_cls):
        # Mock YOLO model instance
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock YOLO results
        mock_result = MagicMock()
        mock_result.boxes.xyxy = torch.tensor([[10.0, 15.0, 40.0, 80.0]])
        mock_result.boxes.conf = torch.tensor([0.89])
        mock_result.boxes.id = torch.tensor([3])
        mock_result.boxes.__len__ = lambda self: 1

        mock_model.track.return_value = [mock_result]

        # Initialize detector - verify model is loaded once
        detector = PersonDetector(model_path="yolov8n.pt", confidence=0.4, track=True)
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        output = detector.detect_frame(frame, frame_id=42, timestamp=1.68)

        # 1. Validate top-level contract keys
        self.assertIn("frame_id", output)
        self.assertIn("timestamp", output)
        self.assertIn("frame", output)
        self.assertIn("annotated_frame", output)
        self.assertIn("people", output)

        self.assertEqual(output["frame_id"], 42)
        self.assertEqual(output["timestamp"], 1.68)
        self.assertIs(output["frame"], frame)
        self.assertIsInstance(output["annotated_frame"], np.ndarray)
        self.assertIsInstance(output["people"], list)

        # 2. Validate person item contract
        self.assertEqual(len(output["people"]), 1)
        person = output["people"][0]
        self.assertEqual(person["track_id"], 3)
        self.assertEqual(person["bbox"], [10, 15, 40, 80])
        self.assertEqual(person["confidence"], 0.89)

    @patch("src.detection.person_detector.YOLO")
    def test_zero_person_frame_handling(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.track.return_value = [mock_result]

        detector = PersonDetector(track=True)
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        output = detector.detect_frame(frame, frame_id=1, timestamp=0.0)

        self.assertEqual(output["people"], [])
        self.assertEqual(output["frame_id"], 1)
        self.assertIsInstance(output["annotated_frame"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
