# Member 2: Safety Event & Pose Analytics Module

This directory contains the implementation of **Member 2: Safety Event Detection** for the Women Safety Analytics CCTV prototype.

---

## 1. Responsibilities & Principles

Member 2 builds directly on top of Member 1's YOLO person detection, tracking IDs, and bounding boxes to detect measurable visual patterns that justify human review:

1. **Predefined Distress Gesture Detection (`potential_distress_gesture`)**:
   - Uses MediaPipe Pose to detect both hands/wrists raised above the shoulders/head region and held for a sustained duration.
   - Evaluates landmark visibility, elevation above shoulders, and temporal persistence.
   - Maintained per persistent tracking ID with cooldown deduplication.
2. **Sustained Proximity & Close-Following Analysis (`sustained_proximity`)**:
   - Uses tracking coordinates (bottom-center ground position) and normalized image-space distance (scaled by average bounding-box height).
   - Computes movement vectors and cosine similarity to ensure correlated motion.
   - Rejects brief path-crossings and stationary standing bystanders.
   - Maintained per canonical tracking pair `(min(id1, id2), max(id1, id2))`.
3. **Safety Event Manager & Unified Public Facade (`SafetyEventDetector`)**:
   - Encapsulates pose estimation, gesture detection, proximity analysis, and deduplication behind a single, clean API for Member 3.

> **CRITICAL RESPONSIBLE AI & PRIVACY NOTICE**:
> - All detections are explicitly classified as **potential safety events requiring human verification**.
> - The system does **NOT** determine whether harassment, stalking, assault, or an emergency has actually occurred.
> - The system does **NOT** perform gender classification, facial recognition, age estimation, or emotion analysis. Tracking IDs are anonymous, temporary session counters.
> - Monocular CCTV cameras do not provide calibrated 3D world distances in meters; proximity is measured in normalized image-space scale.

---

## 2. Module Structure

```
src/safety/
├── __init__.py                # Clean exports of public Member 2 classes and constants
├── pose_detector.py           # Reusable MediaPipe Pose wrapper (loaded once)
├── distress_detector.py       # Predefined distress gesture detector (temporal state & cooldown)
├── proximity_detector.py      # Sustained proximity & correlated movement detector
├── event_manager.py           # Standardized safety event creation & schema validation
├── safety_event_detector.py   # High-level unified facade and visual annotation helper
└── README.md                  # Comprehensive Member 2 documentation
```

---

## 3. Implemented Safety Events & Rules

### Event 1: Predefined Distress Gesture (`potential_distress_gesture`)

*Prototype Definition*: Both hands/wrists intentionally raised above the shoulders/head region and held continuously for a sustained duration.

*Rules & Thresholds*:
- **Landmarks Evaluated**: Left wrist (15), Right wrist (16), Left shoulder (11), Right shoulder (12).
- **Coordinate System**: Normalized image space ($y=0.0$ at top, $1.0$ at bottom). Wrist is above shoulder when $y_{\text{wrist}} < y_{\text{shoulder}}$.
- **Landmark Visibility**: $\min(\text{visibility}) \ge 0.5$ across all 4 key landmarks.
- **Both Hands Required**: If only one hand is raised, no event is generated.
- **Temporal Persistence**: Default $\ge 1.0\,\text{s}$ ($\ge 10$ consecutive frames). Single-frame or noisy detections are rejected.
- **Cooldown**: Default $5.0\,\text{s}$ per tracking ID to suppress duplicate alerts.
- **Heuristic Confidence**: Score in $[0.40, 0.98]$ derived from landmark visibility, wrist elevation, and hold duration.

### Event 2: Sustained Proximity / Close-Following (`sustained_proximity`)

*Prototype Definition*: Two tracked people remain in close normalized proximity while moving through the scene together in correlated directions for a sustained period.

*Rules & Thresholds*:
- **Representative Position**: Bottom-center of bounding box ($x = \frac{x_1 + x_2}{2}, y = y_2$).
- **Normalized Distance**: Euclidean distance divided by average bounding-box height ($\frac{\text{dist}}{\text{avg\_height}}$). Threshold default $\le 1.3$.
- **Active Motion**: Minimum displacement threshold ($\ge 8$ pixels over trajectory window). Excludes stationary people standing still.
- **Movement Direction Consistency**: Cosine similarity between velocity vectors ($\ge 0.40$). Opposite-direction path crossings yield negative similarity and are rejected.
- **Temporal Persistence**: Default $\ge 2.0\,\text{s}$ ($\ge 15$ consecutive frames).
- **Pair Canonicalization**: Track ID pairs are indexed symmetrically as $(\min(id_a, id_b), \max(id_a, id_b))$.
- **Cooldown**: Default $8.0\,\text{s}$ per pair to prevent repeated event emission.
- **Heuristic Confidence**: Score in $[0.35, 0.95]$ derived from normalized proximity closeness, cosine similarity, and sustained duration.

---

## 4. Standard Safety Event Contract (Member 3 Handoff)

Member 2 emits dictionaries strictly conforming to this schema:

```json
{
  "event_type": "potential_distress_gesture",
  "track_ids": [2],
  "timestamp": 12.7,
  "frame_id": 381,
  "camera_id": "CAM_01",
  "confidence": 0.84,
  "status": "pending",
  "details": {
    "gesture_name": "both_hands_raised",
    "held_duration_sec": 1.4,
    "consecutive_frames": 14,
    "wrist_elevation": 0.22,
    "landmark_visibility": 0.91,
    "disclaimer": "Potential distress gesture requiring human verification."
  }
}
```

For sustained proximity between pairs:

```json
{
  "event_type": "sustained_proximity",
  "track_ids": [3, 5],
  "timestamp": 27.4,
  "frame_id": 822,
  "camera_id": "CAM_01",
  "confidence": 0.76,
  "status": "pending",
  "details": {
    "normalized_distance": 0.72,
    "pixel_distance": 84.5,
    "duration_sec": 3.1,
    "consecutive_frames": 31,
    "motion_similarity": 0.88,
    "disclaimer": "Potential sustained proximity event requiring human verification."
  }
}
```

---

## 5. Member 3 Usage Example

Member 3 only needs to consume `SafetyEventDetector`:

```python
from src.detection.person_detector import PersonDetector
from src.safety import SafetyEventDetector

person_detector = PersonDetector(model_path="yolov8n.pt", confidence=0.4, track=True)
safety_detector = SafetyEventDetector(camera_id="CAM_01")

for frame_data in person_detector.process_video("data/v2.mp4"):
    # 1. Detect potential safety events
    events = safety_detector.process_frame(frame_data)
    
    # 2. Get annotated frame (bounding boxes + safety banners)
    annotated_frame = safety_detector.annotate_events(
        frame=frame_data["annotated_frame"],
        events=events,
        people=frame_data["people"]
    )
    
    # Member 3 Streamlit UI:
    # st_image_placeholder.image(annotated_frame, channels="BGR")
    # for event in events:
    #     render_alert_card(event)  # Verify / Dismiss buttons
```

---

## 6. Known Limitations & Edge Cases

1. **Monocular CCTV Perspective**: Bounding box size varies significantly with distance from the camera. Normalized distance mitigates this but cannot substitute for multi-camera 3D calibration.
2. **Crowds & Groups**: Two friends or commuters walking together naturally satisfy proximity and movement similarity, producing a `sustained_proximity` potential event. This is an expected false positive that justifies human review.
3. **Occlusions & Missing Track IDs**: If a tracking ID drops or is reassigned due to occlusion, temporal timers reset safely without crashing.
4. **Arm Gestures**: Non-distress actions (stretching, cheering, adjusting hair with both hands) can trigger the prototype distress gesture.
