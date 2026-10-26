# Member 3 Extension Point: Streamlit Dashboard

This directory is reserved for **Member 3** to build the Streamlit-based monitoring dashboard, alert triage UI, and verification workflow.

---

## 1. Responsibilities for Member 3

1. **Dashboard Interface**:
   - Provide video selection (e.g. choose sample video from `data/normal/` or `data/fall/`, or upload `.mp4`).
   - Display real-time video stream using `st.image()` with `annotated_frame`.
   - Display live metrics: current FPS, total people tracked, current timestamp, camera ID.
2. **Alert Monitoring Panel**:
   - Display incoming safety events (from Member 2's event detector).
   - Display event details: Event Type, Timestamp, Camera ID, Person Track ID, Confidence, Status.
3. **Human Verification Workflow**:
   - Provide **"Verify"** and **"Dismiss"** action buttons for each pending alert.
   - Update event status in the UI and persist to SQLite (see `src/database/README.md`).

---

## 2. Integration with Member 1's Detection Pipeline

Member 1's `PersonDetector` runs in pure headless mode and **never calls `cv2.imshow()`**.  
Member 3 can import and iterate over video frames directly within Streamlit:

```python
import streamlit as st
import cv2
from src.detection import PersonDetector

st.set_page_config(page_title="Women Safety Analytics", layout="wide")
st.title("AI CCTV Safety Monitoring Dashboard")

video_path = st.sidebar.selectbox("Select Video", ["data/normal/walking_01.mp4", "data/fall/fall_01.mp4"])
detector = PersonDetector(confidence=0.4, track=True)

video_placeholder = st.empty()
info_placeholder = st.sidebar.empty()

for frame_data in detector.process_video(video_path):
    frame_id = frame_data["frame_id"]
    timestamp = frame_data["timestamp"]
    annotated_frame = frame_data["annotated_frame"]
    people = frame_data["people"]

    # Convert BGR to RGB for Streamlit rendering
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    info_placeholder.markdown(f"""
    **Frame:** {frame_id}  
    **Timestamp:** {timestamp:.2f}s  
    **People Detected:** {len(people)}  
    """)

    # MEMBER 3: Interleave Member 2's safety check and display alerts here
```
