"""
Main CLI entry point for the Women Safety Analytics detection pipeline (Member 1).

Usage Examples:
    # Run interactive detection on a prerecorded video:
    python main.py --video data/normal/walking_01.mp4

    # Run detection and save the annotated output video:
    python main.py --video data/normal/walking_01.mp4 --save

    # Run detection with custom confidence and headless mode (no GUI window):
    python main.py --video data/fall/fall_01.mp4 --confidence 0.5 --headless

    # Run with webcam input:
    python main.py --video 0
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

from src.detection.person_detector import PersonDetector
from src.detection.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Women Safety Analytics — Member 1 Video Processing & Person Tracking Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file (.mp4, .avi) or camera index (e.g. 0 for webcam)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or identifier (default: 'yolov8n.pt')."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for person detection (default: 0.4)."
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable multi-person tracking and run raw detection only."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video to the outputs/ directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom file path for saving annotated video (overrides default outputs/ filename)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (cv2.imshow). Recommended for headless servers or automated runs."
    )
    return parser.parse_args()


def resolve_output_path(video_source: str, custom_output: str = None) -> Path:
    """Determine the destination path for saving annotated output video."""
    if custom_output:
        out_p = Path(custom_output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        return out_p

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if video_source.isdigit():
        base_name = f"webcam_{video_source}"
    else:
        base_name = Path(video_source).stem

    return outputs_dir / f"annotated_{base_name}.mp4"


def main() -> int:
    args = parse_args()

    # 1. Validate video source path
    is_cam = args.video.isdigit()
    if not is_cam:
        video_file = Path(args.video)
        if not video_file.is_file():
            print(f"[ERROR] Video file not found: {args.video}", file=sys.stderr)
            print("Please check the path and try again. Example: --video data/normal/walking_01.mp4", file=sys.stderr)
            return 1

    print("=" * 60)
    print(" Women Safety Analytics — Detection & Tracking (Member 1)")
    print("=" * 60)
    print(f" Source:     {args.video}")
    print(f" Model:      {args.model}")
    print(f" Confidence: {args.confidence}")
    print(f" Tracking:   {'Enabled' if not args.no_track else 'Disabled'}")
    print(f" Headless:   {args.headless}")
    print(f" Save Video: {args.save}")
    print("=" * 60)

    # 2. Initialize VideoProcessor
    try:
        processor = VideoProcessor(args.video)
    except Exception as e:
        print(f"[ERROR] Failed to open video source: {e}", file=sys.stderr)
        return 1

    print(f"[INFO] Video loaded: {processor.width}x{processor.height} @ {processor.fps:.2f} FPS")
    if processor.total_frames > 0:
        print(f"[INFO] Total frames: {processor.total_frames}")

    # 3. Initialize PersonDetector (Loads YOLO model ONCE)
    print(f"[INFO] Loading YOLO model ({args.model})...")
    try:
        detector = PersonDetector(
            model_path=args.model,
            confidence=args.confidence,
            track=not args.no_track
        )
    except Exception as e:
        processor.release()
        print(f"[ERROR] Failed to load YOLO detector: {e}", file=sys.stderr)
        return 1
    print("[INFO] Model loaded successfully.")

    # 4. Setup VideoWriter if --save is requested
    writer = None
    if args.save:
        out_path = resolve_output_path(args.video, args.output)
        # Use mp4v codec for standard MP4 output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            processor.fps,
            (processor.width, processor.height)
        )
        if not writer.isOpened():
            print(f"[WARNING] Could not open VideoWriter for {out_path}. Video will not be saved.")
            writer = None
        else:
            print(f"[INFO] Saving annotated output to: {out_path}")

    # 5. Process video stream
    window_name = "Women Safety Analytics - Member 1 Detection"
    frame_count = 0
    start_time = time.time()

    try:
        for frame_id, timestamp, frame in processor.iter_frames():
            frame_count += 1

            # Run detection & tracking
            data = detector.detect_frame(frame, frame_id=frame_id, timestamp=timestamp)
            annotated_frame = data["annotated_frame"]
            people = data["people"]

            # Write to disk if saving
            if writer is not None:
                writer.write(annotated_frame)

            # Display frame if not in headless mode
            if not args.headless:
                # Add status banner on top
                banner_text = f"Frame: {frame_id} | Time: {timestamp:.2f}s | People: {len(people)} | 'q' to quit"
                cv2.putText(
                    annotated_frame,
                    banner_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, annotated_frame)

                # Wait for key press (mapped to FPS timing)
                delay_ms = max(1, int(1000 / processor.fps))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\n[INFO] Playback stopped by user.")
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Processed {frame_count} frames ({timestamp:.2f}s) — Detected {len(people)} people")

    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")
    finally:
        # 6. Clean release of all resources
        processor.release()
        if writer is not None:
            writer.release()
            print("[INFO] Output video successfully saved.")
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"\n[DONE] Finished processing {frame_count} frames in {elapsed:.2f}s (Average FPS: {avg_fps:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
