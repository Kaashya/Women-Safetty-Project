# Member 3: Streamlit Dashboard

The dashboard is the UI and persistence consumer for the existing Member 1 and Member 2 pipeline. It does not load or implement another detector, tracker, pose model, or safety heuristic.

## Run

From the repository root, install project requirements and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

Upload an MP4, AVI, MOV, or MKV clip in the sidebar, optionally enter a camera ID, then select **Start Analysis**. Analysis does not start automatically. **Stop / Reset** releases the video and safety resources and clears the current feed state without deleting saved events.

## Per-frame integration

```python
frame_data = person_detector.detect_frame(frame, frame_id=frame_id, timestamp=timestamp)
events = safety_detector.process_frame(frame_data)
annotated_frame = safety_detector.annotate_events(
    frame_data["annotated_frame"],
    events=events,
    people=frame_data["people"],
)
```

This is the public integration used by the app. It displays the BGR annotated image as RGB and persists emitted events through `src/database/database.py`. Streamlit session state retains the model instances and temporal safety detector between fragment updates; a new video constructs fresh instances so per-video tracking and safety state starts clean.

## Operator workflow and event language

- The dashboard displays **Potential Distress Gesture** and **Sustained Proximity** as potential safety events requiring human verification.
- **Verify** changes `pending` to `verified`; this records an operator review and is not proof of a crime.
- **Dismiss** changes `pending` to `dismissed`. Dismissed alerts remain in event history.
- Event scores are heuristic, not calibrated probabilities. Track IDs are temporary labels and do not identify people.
- The Event History table can be filtered by status and event type.

Uploaded video bytes are written to a temporary file because OpenCV opens a file path. The file is released and removed on video completion or reset. The database is local at `data/events.db`; `.db` files are ignored by Git.

Processing is sequential and runs as quickly as the available hardware allows. The dashboard does not require CCTV or RTSP input.
