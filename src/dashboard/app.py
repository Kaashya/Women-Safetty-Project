"""Streamlit application integrating the existing detection and safety modules."""

from pathlib import Path
import sys
import sqlite3
import tempfile
from typing import Any, Optional

import cv2
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from src.dashboard.helpers import (
    event_label,
    format_event_score,
    format_timestamp,
    format_track_ids,
)
from src.database.database import (
    DATABASE_PATH,
    get_event_counts,
    get_events,
    get_pending_events,
    initialize_database,
    insert_event,
    update_event_status,
)
from src.detection import PersonDetector, VideoProcessor
from src.safety import SafetyEventDetector


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
CAMERA_ID_DEFAULT = "CAM_01"
DATABASE_LIMIT = 500
FRAMES_PER_TICK = 5


def _initialize_session_state() -> None:
    defaults = {
        "analysis_started": False,
        "is_processing": False,
        "video_complete": False,
        "last_frame_rgb": None,
        "current_timestamp": 0.0,
        "current_people_count": 0,
        "current_frame_id": 0,
        "source_video": None,
        "active_camera_id": CAMERA_ID_DEFAULT,
        "pose_available": True,
        "processing_error": None,
        "_video_processor": None,
        "_person_detector": None,
        "_safety_detector": None,
        "_temporary_video_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "camera_id_input" not in st.session_state:
        st.session_state.camera_id_input = CAMERA_ID_DEFAULT


def _release_processing_resources() -> None:
    processor = st.session_state.get("_video_processor")
    if processor is not None:
        processor.release()

    safety_detector = st.session_state.get("_safety_detector")
    if safety_detector is not None:
        safety_detector.close()

    temporary_path = st.session_state.get("_temporary_video_path")
    if temporary_path:
        Path(temporary_path).unlink(missing_ok=True)

    st.session_state["_video_processor"] = None
    st.session_state["_person_detector"] = None
    st.session_state["_safety_detector"] = None
    st.session_state["_temporary_video_path"] = None


def _reset_analysis() -> None:
    _release_processing_resources()
    st.session_state.is_processing = False
    st.session_state.video_complete = False
    st.session_state.analysis_started = False
    st.session_state.last_frame_rgb = None
    st.session_state.current_timestamp = 0.0
    st.session_state.current_people_count = 0
    st.session_state.current_frame_id = 0
    st.session_state.source_video = None
    st.session_state.pose_available = True
    st.session_state.processing_error = None


def _start_analysis(uploaded_file: Any, camera_id: str) -> Optional[str]:
    _reset_analysis()
    if uploaded_file is None:
        return "Upload a video before starting analysis."

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        return "Unsupported video format. Choose an MP4, AVI, MOV, or MKV file."

    video_bytes = uploaded_file.getvalue()
    if not video_bytes:
        return "The selected video is empty."

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="women_safety_",
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        st.session_state["_temporary_video_path"] = temporary_path
        processor = VideoProcessor(temporary_path)
        st.session_state["_video_processor"] = processor
        person_detector = PersonDetector(confidence=0.4, track=True)
        st.session_state["_person_detector"] = person_detector
        safety_detector = SafetyEventDetector(camera_id=camera_id)
        st.session_state["_safety_detector"] = safety_detector
    except Exception as error:
        _release_processing_resources()
        return f"Could not start video analysis: {error}"

    st.session_state.is_processing = True
    st.session_state.analysis_started = True
    st.session_state.video_complete = False
    st.session_state.source_video = Path(uploaded_file.name).name
    st.session_state.active_camera_id = camera_id
    st.session_state.current_timestamp = 0.0
    st.session_state.current_people_count = 0
    st.session_state.current_frame_id = 0
    st.session_state.last_frame_rgb = None
    st.session_state.pose_available = safety_detector.pose_detector.pose_model is not None
    st.session_state.processing_error = None
    return None


def _process_next_frame() -> bool:
    """Process one frame through Member 1 then Member 2; return whether EOF was reached."""
    processor = st.session_state["_video_processor"]
    person_detector = st.session_state["_person_detector"]
    safety_detector = st.session_state["_safety_detector"]

    success, frame, frame_id, timestamp = processor.read_frame()
    if not success or frame is None:
        return True

    frame_data = person_detector.detect_frame(
        frame,
        frame_id=frame_id,
        timestamp=timestamp,
    )
    events = safety_detector.process_frame(frame_data)
    annotated_frame = safety_detector.annotate_events(
        frame_data["annotated_frame"],
        events=events,
        people=frame_data["people"],
    )
    st.session_state.last_frame_rgb = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB,
    )
    st.session_state.current_timestamp = frame_data["timestamp"]
    st.session_state.current_people_count = len(frame_data["people"])
    st.session_state.current_frame_id = frame_data["frame_id"]

    for event in events or []:
        persisted_event = dict(event)
        persisted_event.setdefault("camera_id", st.session_state.active_camera_id)
        persisted_event.setdefault("source_video", st.session_state.source_video)
        insert_event(persisted_event, db_path=DATABASE_PATH)
    return False


def _render_video_panel() -> None:
    completed = False
    failed = False
    st.subheader("Video Analysis")
    if st.session_state.is_processing:
        try:
            for _ in range(FRAMES_PER_TICK):
                completed = _process_next_frame()
                if completed:
                    break
            if completed:
                st.session_state.is_processing = False
                _release_processing_resources()
                if st.session_state.current_frame_id == 0:
                    st.session_state.processing_error = (
                        "No readable video frames were found in the uploaded file."
                    )
                    failed = True
                else:
                    st.session_state.video_complete = True
        except Exception as error:
            st.session_state.is_processing = False
            _release_processing_resources()
            st.session_state.processing_error = str(error)
            failed = True

    if st.session_state.last_frame_rgb is not None:
        st.image(
            st.session_state.last_frame_rgb,
            channels="RGB",
            use_column_width=True,
        )
    elif st.session_state.is_processing:
        st.info("Preparing the annotated video feed…")
    else:
        st.info("Upload a video and start analysis to view the annotated feed.")

    if st.session_state.processing_error:
        st.error(
            "Video analysis stopped because of an error: "
            f"{st.session_state.processing_error}"
        )
    elif st.session_state.video_complete:
        st.success("Video analysis complete.")
    elif st.session_state.is_processing:
        st.caption(
            f"Camera: {st.session_state.active_camera_id} · "
            f"Frame: {st.session_state.current_frame_id} · "
            f"Video time: {format_timestamp(st.session_state.current_timestamp)} · "
            f"People: {st.session_state.current_people_count}"
        )

    if completed or failed:
        st.rerun()


def _render_alerts() -> None:
    st.subheader("Active Alerts")
    pending_events = get_pending_events(db_path=DATABASE_PATH)
    if not pending_events:
        st.info("No active safety alerts.")
        return

    for event in pending_events:
        event_id = event["id"]
        with st.container(border=True):
            st.markdown(f"#### ⚠ {event_label(event.get('event_type'))}")
            details = st.columns(4)
            details[0].write(f"**Person ID(s)**  \n{format_track_ids(event.get('track_ids'))}")
            details[1].write(
                f"**Video Time**  \n{format_timestamp(event.get('video_timestamp'))}"
            )
            details[2].write(f"**Camera**  \n{event.get('camera_id') or 'Unknown'}")
            details[3].write(
                f"**Event Score**  \n{format_event_score(event.get('confidence'))}"
            )
            st.caption("Pending · Requires human verification.")
            actions = st.columns([1, 1, 6])
            if actions[0].button("Verify", key=f"verify_{event_id}"):
                update_event_status(event_id, "verified", db_path=DATABASE_PATH)
                st.rerun()
            if actions[1].button("Dismiss", key=f"dismiss_{event_id}"):
                update_event_status(event_id, "dismissed", db_path=DATABASE_PATH)
                st.rerun()


def _render_event_history() -> None:
    st.subheader("Event History")
    status_labels = {
        "All": None,
        "Pending": "pending",
        "Verified": "verified",
        "Dismissed": "dismissed",
    }
    filter_columns = st.columns(2)
    selected_status = filter_columns[0].selectbox(
        "Status",
        list(status_labels),
        key="history_status_filter",
    )
    event_types = [
        ("All", None),
        ("Potential Distress Gesture", "potential_distress_gesture"),
        ("Sustained Proximity", "sustained_proximity"),
    ]
    selected_event_label = filter_columns[1].selectbox(
        "Event type",
        [label for label, _ in event_types],
        key="history_event_filter",
    )
    selected_event_type = dict(event_types)[selected_event_label]
    history = get_events(
        status=status_labels[selected_status],
        event_type=selected_event_type,
        limit=DATABASE_LIMIT,
        db_path=DATABASE_PATH,
    )
    if not history:
        st.info("No events recorded for these filters.")
        return

    rows = [
        {
            "Time": format_timestamp(event.get("video_timestamp")),
            "Event": event_label(event.get("event_type")),
            "Person ID(s)": format_track_ids(event.get("track_ids")),
            "Camera": event.get("camera_id") or "Unknown",
            "Event Score": format_event_score(event.get("confidence")),
            "Status": str(event.get("status", "pending")).title(),
        }
        for event in history
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if len(history) == DATABASE_LIMIT:
        st.caption(f"Showing the newest {DATABASE_LIMIT} matching events.")


def _render_metrics() -> None:
    counts = get_event_counts(db_path=DATABASE_PATH)
    metrics = st.columns(5)
    metrics[0].metric("People in latest frame", st.session_state.current_people_count)
    metrics[1].metric("Pending alerts", counts["pending"])
    metrics[2].metric("Verified alerts", counts["verified"])
    metrics[3].metric("Dismissed alerts", counts["dismissed"])
    metrics[4].metric("Total events", counts["total"])


def _render_dashboard_content() -> None:
    _render_video_panel()
    try:
        _render_metrics()
        _render_alerts()
        _render_event_history()
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not load or update event history: {error}")


def main() -> None:
    st.set_page_config(page_title="Women Safety Analytics", layout="wide")
    _initialize_session_state()
    try:
        initialize_database(DATABASE_PATH)
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not initialize the event history database: {error}")
        st.stop()

    st.title("Women Safety Analytics")
    st.subheader("AI-assisted CCTV safety monitoring prototype")
    st.warning(
        "Detected events are potential safety events and require human verification. "
        "This prototype does not confirm crimes or identify people."
    )

    with st.sidebar:
        st.header("Video Input")
        uploaded_file = st.file_uploader(
            "Upload a prerecorded video",
            type=["mp4", "avi", "mov", "mkv"],
            disabled=st.session_state.is_processing,
            help="The uploaded file is processed locally and removed from the temporary directory afterward.",
        )
        camera_id = st.text_input(
            "Camera ID",
            key="camera_id_input",
            disabled=st.session_state.is_processing,
            help="This identifier is attached to each event generated during the video analysis.",
        ).strip()
        camera_id = camera_id or CAMERA_ID_DEFAULT
        start_clicked = st.button(
            "Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
        )
        reset_clicked = st.button(
            "Stop / Reset",
            use_container_width=True,
            disabled=not st.session_state.analysis_started,
        )
        st.caption("Inference speed depends on the available hardware.")

    if reset_clicked:
        _reset_analysis()
        st.rerun()

    if start_clicked:
        error_message = _start_analysis(uploaded_file, camera_id)
        if error_message:
            st.error(error_message)
        else:
            st.rerun()

    if st.session_state.analysis_started:
        st.caption(
            f"Current video: {st.session_state.source_video or 'Unknown'} · "
            f"Camera: {st.session_state.active_camera_id}"
        )
        if not st.session_state.pose_available:
            st.warning(
                "Pose estimation could not be initialized. Proximity analysis remains "
                "available, but distress-gesture analysis may not produce events."
            )

    st.caption(
        "Event Score is a heuristic score, not a calibrated probability. "
        "Tracking IDs are temporary and do not identify real-world people."
    )

    fragment_interval = 0.1 if st.session_state.is_processing else None

    @st.fragment(run_every=fragment_interval)
    def dashboard_content() -> None:
        _render_dashboard_content()

    dashboard_content()


if __name__ == "__main__":
    main()
