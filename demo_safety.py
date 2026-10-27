"""
Demonstration CLI application for Member 2 Safety Analytics.

Integrates Member 1's video processing and YOLO tracking with Member 2's
MediaPipe Pose estimation, distress gesture detection, and sustained proximity analytics.

Usage Examples:
    # Run interactive detection with visual window on video:
    python demo_safety.py --video data/18156284-hd_1080_1920_25fps.mp4

    # Run in headless mode and save annotated output video:
    python demo_safety.py --video data/18156284-hd_1080_1920_25fps.mp4 --headless --save

    # Run with custom parameters:
    python demo_safety.py --video data/18156284-hd_1080_1920_25fps.mp4 --distress-hold 1.0 --proximity-duration 2.0
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional
import cv2

from src.detection.person_detector import PersonDetector
from src.detection.video_processor import VideoProcessor
from src.safety.safety_event_detector import SafetyEventDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Women Safety Analytics — Member 2 Safety Event Analytics Demo"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file or camera index (e.g. 0 for webcam)."
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
        help="YOLO detection confidence threshold (default: 0.4)."
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM_01",
        help="CCTV camera identifier (default: 'CAM_01')."
    )
    parser.add_argument(
        "--distress-hold",
        type=float,
        default=1.0,
        help="Minimum seconds hands must remain raised for distress event (default: 1.0s)."
    )
    parser.add_argument(
        "--proximity-dist",
        type=float,
        default=1.3,
        help="Normalized distance threshold in units of average bbox height (default: 1.3)."
    )
    parser.add_argument(
        "--proximity-duration",
        type=float,
        default=2.0,
        help="Minimum seconds sustained proximity must persist (default: 2.0s)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video with safety events to outputs/ directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom destination path for saved video (overrides default outputs/ naming)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI display window (recommended for automated runs / servers)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before stopping (useful for quick testing)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed diagnostic logging."
    )
    return parser.parse_args()


def resolve_output_path(video_source: str, custom_output: Optional[str] = None) -> Path:
    """Determine output video file path."""
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

    return outputs_dir / f"safety_{base_name}.mp4"


def main() -> int:
    args = parse_args()

    # 1. Validate input source
    is_cam = args.video.isdigit()
    if not is_cam:
        video_file = Path(args.video)
        if not video_file.is_file():
            print(f"[ERROR] Video file not found: {args.video}", file=sys.stderr)
            return 1

    print("=" * 68)
    print(" Women Safety Analytics — Integrated Safety Event Demo (Member 2)")
    print("=" * 68)
    print(f" Source:              {args.video}")
    print(f" Camera ID:           {args.camera_id}")
    print(f" YOLO Model:          {args.model} (conf: {args.confidence})")
    print(f" Distress Hold:       {args.distress_hold}s")
    print(f" Proximity Threshold: {args.proximity_dist}x bbox height, {args.proximity_duration}s hold")
    print(f" Headless:            {args.headless}")
    print(f" Save Annotated:      {args.save}")
    print("=" * 68)

    # 2. Open Video Stream via Member 1's VideoProcessor
    try:
        processor = VideoProcessor(args.video)
    except Exception as e:
        print(f"[ERROR] Failed to open video source: {e}", file=sys.stderr)
        return 1

    print(f"[INFO] Video loaded: {processor.width}x{processor.height} @ {processor.fps:.2f} FPS ({processor.total_frames} frames)")

    # 3. Initialize Member 1 PersonDetector (loads YOLO ONCE)
    print(f"[INFO] Initializing YOLO PersonDetector ({args.model})...")
    try:
        person_detector = PersonDetector(
            model_path=args.model,
            confidence=args.confidence,
            track=True
        )
    except Exception as e:
        processor.release()
        print(f"[ERROR] Failed to initialize PersonDetector: {e}", file=sys.stderr)
        return 1

    # 4. Initialize Member 2 SafetyEventDetector (loads MediaPipe Pose ONCE)
    print("[INFO] Initializing Member 2 SafetyEventDetector...")
    try:
        safety_detector = SafetyEventDetector(
            camera_id=args.camera_id,
            distress_hold_sec=args.distress_hold,
            proximity_dist_thresh=args.proximity_dist,
            proximity_duration_sec=args.proximity_duration,
            debug=args.debug
        )
    except Exception as e:
        processor.release()
        print(f"[ERROR] Failed to initialize SafetyEventDetector: {e}", file=sys.stderr)
        return 1

    # 5. Setup VideoWriter if saving is enabled
    writer = None
    if args.save:
        out_path = resolve_output_path(args.video, args.output)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            processor.fps,
            (processor.width, processor.height)
        )
        if not writer.isOpened():
            print(f"[WARNING] Could not open VideoWriter for {out_path}.")
            writer = None
        else:
            print(f"[INFO] Saving annotated output to: {out_path}")

    # 6. Stream and Process Video
    window_name = "Women Safety Analytics - Member 2 Safety Demo"
    frame_count = 0
    total_events_generated = 0
    start_time = time.time()

    try:
        for frame_id, timestamp, frame in processor.iter_frames():
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                print(f"\n[INFO] Reached max frames limit ({args.max_frames}). Stopping.")
                break

            # A. Run Member 1 person detection and tracking
            frame_data = person_detector.detect_frame(
                frame,
                frame_id=frame_id,
                timestamp=timestamp
            )

            # B. Run Member 2 safety event analytics
            events = safety_detector.process_frame(frame_data)
            if events:
                total_events_generated += len(events)
                for ev in events:
                    lbl = safety_detector.event_manager.format_event_label(ev)
                    print(f"\n[POTENTIAL SAFETY EVENT DETECTED @ {timestamp:.2f}s | Frame {frame_id}]")
                    print(f"  -> {lbl}")
                    print(f"  -> Details: {ev.get('details', {})}")

            # C. Produce combined annotated frame (Preserves Member 1's boxes and overlays Member 2's alerts)
            combined_frame = safety_detector.annotate_events(
                frame=frame_data["annotated_frame"],
                events=events,
                people=frame_data["people"]
            )

            # D. Write to disk if enabled
            if writer is not None:
                writer.write(combined_frame)

            # E. Display frame if GUI window is active
            if not args.headless:
                # Add status text
                status_text = f"Frame: {frame_id} | Time: {timestamp:.2f}s | People: {len(frame_data['people'])} | Events: {total_events_generated} | 'q' to quit"
                cv2.putText(
                    combined_frame,
                    status_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, combined_frame)

                delay_ms = max(1, int(1000 / processor.fps))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] Playback stopped by user.")
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Frame {frame_count} ({timestamp:.2f}s) — People: {len(frame_data['people'])} — Total Events: {total_events_generated}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        processor.release()
        safety_detector.close()
        if writer is not None:
            writer.release()
            print("[INFO] Annotated video output successfully saved.")
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 68)
    print(f" Processed: {frame_count} frames in {elapsed:.2f}s (Average FPS: {avg_fps:.2f})")
    print(f" Total Potential Safety Events Detected: {total_events_generated}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
