# Member 2 Extension Point: Safety Event & Pose Detection

This directory is reserved for **Member 2** to implement AI safety event detection (MediaPipe Pose estimation and possible-fall detection).

---

## 1. Responsibilities for Member 2

1. **Pose Estimation**:
   - Utilize MediaPipe Pose to detect body landmarks (shoulders, hips, knees, ankles) on cropped person bounding boxes.
2. **Possible-Fall Detection**:
   - Detect rapid downward motion and horizontal posture transition over consecutive frames for a specific `track_id`.
   - Maintain a temporal buffer of landmarks or bounding-box aspect ratios (e.g. width > height).
3. **Safety Event Generation**:
   - Emit a structured safety event whenever fall criteria are met.
4. **Distress Gesture (Optional / If Time Permits)**:
   - Identify raised hands, waving, or crouched distress posture.

> **CRITICAL: DO NOT RE-IMPLEMENT PERSON DETECTION OR TRACKING.**  
> Member 1's pipeline already provides detected people, bounding boxes, tracking IDs, and a safe cropping helper.

---

## 2. Integration with Member 1's Pipeline

Member 2 imports `PersonDetector` and the `get_person_crop` utility:

```python
from src.detection import PersonDetector
from src.utils.helpers import get_person_crop

# 1. Initialize detector (loads YOLO once)
detector = PersonDetector(model_path="yolov8n.pt", confidence=0.4, track=True)

# 2. Stream video frames
for frame_data in detector.process_video("data/fall/fall_01.mp4"):
    frame = frame_data["frame"]
    timestamp = frame_data["timestamp"]
    frame_id = frame_data["frame_id"]

    for person in frame_data["people"]:
        track_id = person["track_id"]
        bbox = person["bbox"]          # [x1, y1, x2, y2]
        confidence = person["confidence"]

        # Safely extract cropped person image (bounds checked and clamped)
        crop = get_person_crop(frame, bbox)
        if crop.size == 0:
            continue

        # -------------------------------------------------------------
        # MEMBER 2 TODO:
        # - Pass `crop` to MediaPipe Pose
        # - Track pose landmarks per `track_id` over time
        # - Check for sudden vertical drop or horizontal orientation
        # - Emit safety event if fall condition is met
        # -------------------------------------------------------------
```

---

## 3. Standard Safety Event Contract

Member 2 must emit safety events conforming to this dictionary structure so Member 3 can consume them in the dashboard and database:

```json
{
  "event_type": "possible_fall",
  "track_id": 3,
  "timestamp": 12.7,
  "camera_id": "CAM_01",
  "confidence": 0.81,
  "status": "pending"
}
```

### Event Fields:
- `event_type`: (str) Event description, e.g. `"possible_fall"` or `"distress_gesture"`.
- `track_id`: (int or None) Tracking ID of the person associated with the event.
- `timestamp`: (float) Video timestamp in seconds when the event triggered.
- `camera_id`: (str) Identifier for the camera stream (default: `"CAM_01"`).
- `confidence`: (float) Confidence score of the safety event classifier (0.0 to 1.0).
- `status`: (str) Initial triage status, defaults to `"pending"` (to be updated to `"verified"` or `"dismissed"` by human operator in Member 3's dashboard).
