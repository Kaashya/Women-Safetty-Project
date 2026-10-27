"""
Sustained Proximity and Close-Following Detection Module.

Heuristic Purpose:
    Identifies pairs of tracked people who:
    1. Remain in close image-space proximity normalized by body scale.
    2. Move through the scene together in correlated directions for a sustained duration.
    3. Exclude brief path-crossings and stationary bystanders.

IMPORTANT DISCLAIMER:
    This heuristic detects measurable visual patterns of spatial/temporal proximity.
    It does NOT determine or imply harassment, stalking, assault, or intent.
    Detections are potential safety events that require human review and contextual evaluation.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.safety.event_manager import EVENT_SUSTAINED_PROXIMITY, SafetyEventManager


@dataclass
class PersonMotionState:
    """Recent trajectory and position history for a tracked person."""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # (timestamp, bottom_x, bottom_y)
    last_bbox_h: float = 100.0
    last_bbox_w: float = 50.0
    last_seen_ts: float = 0.0


@dataclass
class PairProximityState:
    """Temporal proximity and motion correlation state for a pair of tracked persons."""
    pair: Tuple[int, int]
    first_close_ts: Optional[float] = None
    last_close_ts: Optional[float] = None
    consecutive_close_frames: int = 0
    recent_similarities: deque = field(default_factory=lambda: deque(maxlen=30))
    last_event_ts: Optional[float] = None
    last_seen_ts: float = 0.0


class ProximityDetector:
    """
    Detects sustained proximity and correlated movement between pairs of tracked persons.

    Employs normalized distance heuristics (scaled by average bounding-box height)
    and cosine similarity of movement vectors to differentiate genuine following/proximity
    from brief crossings and stationary standing.
    """

    def __init__(
        self,
        event_manager: Optional[SafetyEventManager] = None,
        proximity_threshold: float = 1.3,
        min_duration_sec: float = 2.0,
        min_consecutive_frames: int = 15,
        min_movement_speed: float = 8.0,
        min_movement_similarity: float = 0.4,
        cooldown_sec: float = 8.0,
        stale_timeout_sec: float = 4.0
    ) -> None:
        """
        Args:
            event_manager: SafetyEventManager instance for standardized event output.
            proximity_threshold: Maximum normalized distance (units of average bbox height, default: 1.3).
            min_duration_sec: Minimum seconds the proximity condition must persist (default: 2.0s).
            min_consecutive_frames: Minimum consecutive frames required (default: 15 frames).
            min_movement_speed: Minimum pixel displacement over the motion window to classify as moving.
            min_movement_similarity: Minimum cosine similarity between movement vectors (default: 0.4).
            cooldown_sec: Seconds to suppress duplicate alerts for the same pair (default: 8.0s).
            stale_timeout_sec: Seconds of inactivity before pruning tracking history (default: 4.0s).
        """
        self.event_manager = event_manager or SafetyEventManager()
        self.proximity_threshold = float(proximity_threshold)
        self.min_duration_sec = float(min_duration_sec)
        self.min_consecutive_frames = int(min_consecutive_frames)
        self.min_movement_speed = float(min_movement_speed)
        self.min_movement_similarity = float(min_movement_similarity)
        self.cooldown_sec = float(cooldown_sec)
        self.stale_timeout_sec = float(stale_timeout_sec)

        # Person motion states: Dict[int, PersonMotionState]
        self.person_states: Dict[int, PersonMotionState] = {}

        # Canonical pair states: Dict[Tuple[int, int], PairProximityState]
        self.pair_states: Dict[Tuple[int, int], PairProximityState] = {}

    @staticmethod
    def get_bottom_center(bbox: List[int]) -> Tuple[float, float, float, float]:
        """
        Extract representative ground-plane position and dimensions from a bounding box.

        Returns:
            Tuple of (bottom_center_x, bottom_y, width, height).
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        by = float(y2)
        w = float(x2 - x1)
        h = float(y2 - y1)
        return cx, by, w, h

    @staticmethod
    def canonical_pair_key(id_a: int, id_b: int) -> Tuple[int, int]:
        """Ensure symmetric representation: (1, 2) and (2, 1) map to (1, 2)."""
        return (min(id_a, id_b), max(id_a, id_b))

    def _compute_movement_vector(
        self,
        motion_state: PersonMotionState,
        window_sec: float = 0.8
    ) -> Tuple[float, float, float]:
        """
        Compute movement displacement vector (dx, dy) and speed over recent history.

        Returns:
            Tuple of (dx, dy, speed).
        """
        history = motion_state.positions
        if len(history) < 2:
            return 0.0, 0.0, 0.0

        curr_ts, curr_x, curr_y = history[-1]

        # Find position closest to window_sec in the past
        target_ts = curr_ts - window_sec
        prev_x, prev_y = history[0][1], history[0][2]

        for ts, px, py in reversed(history):
            if ts <= target_ts:
                prev_x, prev_y = px, py
                break

        dx = curr_x - prev_x
        dy = curr_y - prev_y
        speed = math.hypot(dx, dy)
        return dx, dy, speed

    def _compute_motion_similarity(
        self,
        vec_a: Tuple[float, float, float],
        vec_b: Tuple[float, float, float]
    ) -> Tuple[float, bool]:
        """
        Compute cosine similarity between two movement vectors.

        Returns:
            Tuple of (similarity: float [-1.0, 1.0], are_moving: bool).
        """
        dx_a, dy_a, speed_a = vec_a
        dx_b, dy_b, speed_b = vec_b

        # Check if both persons exhibit sufficient active movement
        are_moving = (speed_a >= self.min_movement_speed) and (speed_b >= self.min_movement_speed)
        if not are_moving:
            return 0.0, False

        dot_product = (dx_a * dx_b) + (dy_a * dy_b)
        cosine_sim = dot_product / (speed_a * speed_b)
        # Numerical clamping to [-1.0, 1.0]
        clamped_sim = float(np.clip(cosine_sim, -1.0, 1.0))
        return clamped_sim, True

    def update(
        self,
        people: List[Dict[str, Any]],
        timestamp: float,
        frame_id: int
    ) -> List[Dict[str, Any]]:
        """
        Update temporal tracking pairs and detect sustained proximity / following events.

        Args:
            people: Detected people list with 'track_id' and 'bbox'.
            timestamp: Video timestamp in seconds.
            frame_id: Frame sequence number.

        Returns:
            List of generated sustained proximity safety events.
        """
        events: List[Dict[str, Any]] = []

        # 1. Update individual motion histories (only for people with valid tracking IDs)
        active_people: List[Dict[str, Any]] = []
        for person in people:
            tid = person.get("track_id")
            bbox = person.get("bbox")
            if tid is None or bbox is None:
                continue

            cx, by, w, h = self.get_bottom_center(bbox)

            if tid not in self.person_states:
                self.person_states[tid] = PersonMotionState(track_id=tid)
            pstate = self.person_states[tid]
            pstate.positions.append((timestamp, cx, by))
            pstate.last_bbox_w = w
            pstate.last_bbox_h = h
            pstate.last_seen_ts = timestamp

            active_people.append({
                "track_id": tid,
                "cx": cx,
                "by": by,
                "w": w,
                "h": h,
                "bbox": bbox
            })

        # 2. Evaluate all pairs of tracked people
        num_active = len(active_people)
        evaluated_pairs: Set[Tuple[int, int]] = set()

        for i in range(num_active):
            for j in range(i + 1, num_active):
                p1 = active_people[i]
                p2 = active_people[j]
                id1, id2 = p1["track_id"], p2["track_id"]
                pair_key = self.canonical_pair_key(id1, id2)
                evaluated_pairs.add(pair_key)

                # Compute Euclidean distance between bottom-centers
                pixel_dist = math.hypot(p1["cx"] - p2["cx"], p1["by"] - p2["by"])

                # Normalize by average bounding-box height
                avg_height = max(1.0, (p1["h"] + p2["h"]) / 2.0)
                norm_dist = pixel_dist / avg_height

                # Retrieve motion vectors
                vec1 = self._compute_movement_vector(self.person_states[id1])
                vec2 = self._compute_movement_vector(self.person_states[id2])
                motion_sim, are_moving = self._compute_motion_similarity(vec1, vec2)

                # Initialize or get pair state
                if pair_key not in self.pair_states:
                    self.pair_states[pair_key] = PairProximityState(pair=pair_key)
                pair_state = self.pair_states[pair_key]
                pair_state.last_seen_ts = timestamp

                # Core proximity & movement criteria:
                # 1. Normalized distance <= proximity_threshold
                # 2. Both persons are moving actively
                # 3. Motion directions are similar (cosine similarity >= min_similarity)
                #    (Excludes opposite-direction path crossing where cosine_sim < 0)
                is_close = norm_dist <= self.proximity_threshold
                is_correlated = are_moving and (motion_sim >= self.min_movement_similarity)

                if is_close and is_correlated:
                    pair_state.recent_similarities.append(motion_sim)
                    if pair_state.first_close_ts is None:
                        pair_state.first_close_ts = timestamp
                        pair_state.consecutive_close_frames = 1
                    else:
                        pair_state.consecutive_close_frames += 1

                    pair_state.last_close_ts = timestamp
                    close_duration = timestamp - pair_state.first_close_ts

                    # Check temporal persistence requirements
                    duration_met = close_duration >= self.min_duration_sec
                    frames_met = pair_state.consecutive_close_frames >= self.min_consecutive_frames

                    if duration_met and frames_met:
                        # Check cooldown
                        in_cooldown = (
                            pair_state.last_event_ts is not None
                            and (timestamp - pair_state.last_event_ts) < self.cooldown_sec
                        )

                        if not in_cooldown:
                            # Heuristic confidence score
                            avg_sim = float(np.mean(pair_state.recent_similarities)) if pair_state.recent_similarities else motion_sim
                            dist_score = max(0.0, 1.0 - (norm_dist / self.proximity_threshold))
                            sim_score = max(0.0, avg_sim)
                            dur_score = min(1.0, close_duration / (self.min_duration_sec * 2.0))

                            heur_conf = float(np.clip(
                                0.4 * dist_score + 0.3 * sim_score + 0.3 * dur_score,
                                0.35,
                                0.95
                            ))

                            event = self.event_manager.create_event(
                                event_type=EVENT_SUSTAINED_PROXIMITY,
                                track_ids=[pair_key[0], pair_key[1]],
                                timestamp=timestamp,
                                frame_id=frame_id,
                                confidence=heur_conf,
                                details={
                                    "normalized_distance": round(norm_dist, 2),
                                    "pixel_distance": round(pixel_dist, 1),
                                    "duration_sec": round(close_duration, 2),
                                    "consecutive_frames": pair_state.consecutive_close_frames,
                                    "motion_similarity": round(avg_sim, 2),
                                    "disclaimer": "Potential sustained proximity event requiring human verification."
                                }
                            )
                            events.append(event)
                            pair_state.last_event_ts = timestamp
                else:
                    # Reset duration if proximity or correlated movement breaks
                    # Allow a small grace period or reset directly
                    pair_state.first_close_ts = None
                    pair_state.consecutive_close_frames = 0
                    pair_state.recent_similarities.clear()

        # 3. Clean up stale individual and pair states
        self._prune_stale_states(timestamp)

        return events

    def _prune_stale_states(self, current_ts: float) -> None:
        """Prune tracking histories for people and pairs that haven't been seen recently."""
        # Prune inactive persons
        stale_persons = [
            tid for tid, pstate in self.person_states.items()
            if (current_ts - pstate.last_seen_ts) > self.stale_timeout_sec
        ]
        for tid in stale_persons:
            del self.person_states[tid]

        # Prune pairs involving pruned or stale persons
        stale_pairs = [
            pair for pair, pr_state in self.pair_states.items()
            if (current_ts - pr_state.last_seen_ts) > self.stale_timeout_sec
            or pair[0] in stale_persons
            or pair[1] in stale_persons
        ]
        for pair in stale_pairs:
            del self.pair_states[pair]
