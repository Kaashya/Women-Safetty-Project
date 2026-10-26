"""
Reusable video processing module for reading prerecorded videos or camera streams.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoProcessor:
    """
    OpenCV-based video reader for prerecorded video files and camera inputs.

    Features:
        - Validates video file existence before attempting OpenCV capture.
        - Gracefully handles corrupt or unreadable files.
        - Provides video metadata (FPS, width, height, total frames).
        - Tracks current frame count (1-indexed) and elapsed timestamp in seconds.
        - Implements context management for clean resource cleanup.
        - Provides both single-frame reading and an iterable generator.
    """

    def __init__(self, source: Union[str, int, Path]) -> None:
        """
        Initialize the VideoProcessor.

        Args:
            source: File path to a video (.mp4, .avi, etc.) or an integer camera index.

        Raises:
            FileNotFoundError: If a file path is provided but does not exist.
            ValueError: If OpenCV fails to open the video source or source is invalid.
        """
        self.source = source
        self.is_camera = False
        self.cap = None
        self._is_released = True
        self.frame_count = 0

        # Validate path if string or Path
        if isinstance(source, (str, Path)):
            src_str = str(source).strip()
            # Check if source is a digit string representing camera index (e.g. "0")
            if src_str.isdigit():
                self.source = int(src_str)
                self.is_camera = True
            else:
                path = Path(source)
                if not path.is_file():
                    raise FileNotFoundError(f"Video file not found: {source}")
                self.source = str(path)
        elif isinstance(source, int):
            self.is_camera = True
        else:
            raise ValueError(f"Invalid video source type: {type(source)}")

        # Initialize OpenCV capture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.release()
            raise ValueError(f"OpenCV could not open video source: {self.source}")

        self._is_released = False

        # Extract metadata
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(raw_fps) if raw_fps and raw_fps > 0 and not np.isnan(raw_fps) else 25.0

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else -1

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Read the next frame from the video stream.

        Returns:
            Tuple of:
                - success (bool): True if frame was read successfully, False on EOF/error.
                - frame (np.ndarray or None): BGR image matrix.
                - frame_id (int): 1-indexed sequential frame number.
                - timestamp (float): Elapsed video timestamp in seconds.
        """
        if self._is_released or not self.cap.isOpened():
            return False, None, self.frame_count, 0.0

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, self.frame_count, 0.0

        self.frame_count += 1
        frame_id = self.frame_count

        # Get OpenCV hardware timestamp if available, else calculate from frame count & FPS
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is not None and pos_msec > 0:
            timestamp = pos_msec / 1000.0
        else:
            timestamp = (frame_id - 1) / self.fps

        return True, frame, frame_id, round(timestamp, 3)

    def iter_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Generator yielding frames sequentially until the end of the video.

        Yields:
            Tuple of (frame_id, timestamp, frame)
        """
        while True:
            ret, frame, frame_id, timestamp = self.read_frame()
            if not ret or frame is None:
                break
            yield frame_id, timestamp, frame

    def __iter__(self):
        return self.iter_frames()

    def release(self) -> None:
        """Release the OpenCV video capture resource safely."""
        if not self._is_released and hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self._is_released = True

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
