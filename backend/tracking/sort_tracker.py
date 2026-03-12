"""
SORT – Simple Online and Realtime Tracking
==========================================
Reference: Bewley et al., 2016  (https://arxiv.org/abs/1602.00763)

This module provides two public classes:

  KalmanBoxTracker   – one Kalman filter that follows a single vehicle.
  SortTracker        – manages a pool of KalmanBoxTrackers, runs
                       IoU-based Hungarian matching each frame, and hands
                       back tracked bounding boxes with persistent IDs.

Only numpy and scipy are required (both ship with most Data Science /
OpenCV Python environments).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_to_z(bbox):
    """
    Convert a bounding box [x1, y1, x2, y2] into the Kalman measurement
    vector [cx, cy, s, r].

      cx, cy  – centre of the box
      s       – area  (scale)
      r       – aspect ratio  w / h  (assumed ~constant while tracking)
    """
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s  = w * h
    r  = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r], dtype=float)


def _z_to_bbox(z):
    """
    Convert a Kalman state / measurement [cx, cy, s, r] back to
    [x1, y1, x2, y2].
    """
    # Recover w and h from area s and ratio r:  s = w*h,  r = w/h
    w  = np.sqrt(max(z[2] * z[3], 0.0))
    h  = z[2] / w if w > 0 else 0.0
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


def _iou(bb_a, bb_b):
    """
    Compute Intersection-over-Union between two boxes [x1, y1, x2, y2].
    Returns a float in [0, 1].
    """
    ix1 = max(bb_a[0], bb_b[0])
    iy1 = max(bb_a[1], bb_b[1])
    ix2 = min(bb_a[2], bb_b[2])
    iy2 = min(bb_a[3], bb_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    area_b = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
    union   = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter tracker for one vehicle
# ─────────────────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single vehicle using a 7-dimensional Kalman filter.

    State vector  x  (7 × 1):  [cx, cy, s, r, vx, vy, vs]
    Measurement   z  (4 × 1):  [cx, cy, s, r]

    The aspect ratio r is treated as constant (no velocity term).
    vx, vy, vs are velocities for the centre and area.
    """

    # Class-level counter so every new tracker gets a unique ID.
    _count = 0

    def __init__(self, bbox):
        """
        Args:
            bbox: initial bounding box as [x1, y1, x2, y2].
        """
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # ── Kalman filter matrices ────────────────────────────────────────────
        #
        # State dimension   n = 7
        # Measurement dim   m = 4

        n, m = 7, 4

        # State transition F  (constant-velocity model, dt = 1 frame)
        #
        #   cx'  = cx + vx
        #   cy'  = cy + vy
        #   s'   = s  + vs
        #   r'   = r               (no velocity)
        #   vx'  = vx
        #   vy'  = vy
        #   vs'  = vs
        self.F = np.eye(n)
        self.F[0, 4] = 1   # cx += vx
        self.F[1, 5] = 1   # cy += vy
        self.F[2, 6] = 1   # s  += vs

        # Measurement matrix H  (we observe cx, cy, s, r directly)
        self.H = np.zeros((m, n))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Process noise Q  (how much we trust the motion model)
        # Velocities are noisier than positions, so give them higher variance.
        self.Q = np.diag([1., 1., 10., 1., 0.01, 0.01, 0.0001])

        # Measurement noise R  (sensor / detector noise)
        self.R = np.diag([1., 1., 10., 1.])

        # Initial state covariance P
        # High uncertainty in the velocity components at birth.
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])

        # Initial state estimate x: positions from bbox, velocities = 0
        z0 = _bbox_to_z(bbox)
        self.x = np.zeros((n, 1))
        self.x[:4, 0] = z0

        # Book-keeping
        self.time_since_update = 0   # frames since last detection match
        self.hit_streak         = 0   # consecutive frames with a match
        self.age                = 0   # total frames alive

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self):
        """
        Advance the state by one frame using the motion model.
        Returns the predicted bounding box [x1, y1, x2, y2].
        """
        # Clamp area so it never goes negative (can happen for tiny, noisy blobs).
        if self.x[2] < 0:
            self.x[2] = 0.0

        # x = F @ x
        self.x = self.F @ self.x

        # P = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age              += 1
        self.time_since_update += 1

        return _z_to_bbox(self.x[:4, 0])

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, bbox):
        """
        Correct the state using a new detection.

        Args:
            bbox: matched detection as [x1, y1, x2, y2].
        """
        z = _bbox_to_z(bbox).reshape(4, 1)

        # Innovation  y = z - H @ x
        y = z - self.H @ self.x

        # Innovation covariance  S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain  K = P @ H^T @ S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state  x = x + K @ y
        self.x = self.x + K @ y

        # Updated covariance  P = (I - K @ H) @ P
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update  = 0
        self.hit_streak        += 1

    # ── Read current estimate ─────────────────────────────────────────────────

    def get_bbox(self):
        """Return the current estimated bounding box [x1, y1, x2, y2]."""
        return _z_to_bbox(self.x[:4, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Association helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hungarian_match(detections, predictions, iou_threshold=0.3):
    """
    Match detections to predictions with the Hungarian algorithm.

    Args:
        detections  (N × 4 array): detected boxes [x1, y1, x2, y2]
        predictions (M × 4 array): predicted boxes [x1, y1, x2, y2]
        iou_threshold (float): minimum IoU for a valid match

    Returns:
        matches         – list of (det_idx, pred_idx) pairs
        unmatched_dets  – indices of detections with no match
        unmatched_preds – indices of predictions with no match
    """
    n_det  = len(detections)
    n_pred = len(predictions)

    if n_pred == 0:
        return [], list(range(n_det)), []

    if n_det == 0:
        return [], [], list(range(n_pred))

    # Build IoU matrix  (n_det × n_pred)
    iou_matrix = np.zeros((n_det, n_pred), dtype=float)
    for d, det in enumerate(detections):
        for p, pred in enumerate(predictions):
            iou_matrix[d, p] = _iou(det, pred)

    # Hungarian algorithm minimises cost → use 1 - IoU as cost
    row_idx, col_idx = linear_sum_assignment(1 - iou_matrix)

    matches, unmatched_dets, unmatched_preds = [], [], []

    matched_det_set  = set()
    matched_pred_set = set()

    for d, p in zip(row_idx, col_idx):
        if iou_matrix[d, p] >= iou_threshold:
            matches.append((d, p))
            matched_det_set.add(d)
            matched_pred_set.add(p)

    for d in range(n_det):
        if d not in matched_det_set:
            unmatched_dets.append(d)

    for p in range(n_pred):
        if p not in matched_pred_set:
            unmatched_preds.append(p)

    return matches, unmatched_dets, unmatched_preds


# ─────────────────────────────────────────────────────────────────────────────
# Main SORT tracker – manages the pool of per-vehicle Kalman filters
# ─────────────────────────────────────────────────────────────────────────────

class SortTracker:
    """
    SORT tracker: call ``update()`` once per frame with the detector's
    bounding boxes and receive back boxes with persistent vehicle IDs.

    Parameters
    ----------
    max_age : int
        How many consecutive frames a track can go unmatched before it is
        deleted.  Increase this if vehicles are frequently occluded.
    min_hits : int
        Minimum number of consecutive detections before a track is
        reported.  Set to 1 to show IDs immediately (good for traffic
        cams where detections are reliable).
    iou_threshold : float
        Minimum IoU required to link a detection to an existing track.
    """

    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections):
        """
        Run one step of SORT.

        Args:
            detections: list of detections from the detector, each as
                        [x1, y1, x2, y2, score]  (score is ignored internally
                        but keeping it makes the interface compatible with
                        standard object-detection pipelines).

        Returns:
            List of tuples  (x1, y1, x2, y2, vehicle_id)  for every
            currently active and confirmed track.
        """
        self.frame_count += 1

        # ── Extract [x1, y1, x2, y2] — drop the score column if present ─────────
        det_boxes = np.array(
            [[d[0], d[1], d[2], d[3]] for d in detections],
            dtype=float
        ) if detections else np.empty((0, 4))

        # ── Step 1: Predict new locations for all existing tracks ─────────────
        pred_boxes = np.array(
            [t.predict() for t in self.trackers],
            dtype=float
        ) if self.trackers else np.empty((0, 4))

        # ── Step 2: Match detections ↔ predictions ────────────────────────────
        matches, unmatched_dets, unmatched_preds = _hungarian_match(
            det_boxes, pred_boxes, self.iou_threshold
        )

        # ── Step 3: Update matched tracks with their detection ─────────────────
        for d_idx, p_idx in matches:
            self.trackers[p_idx].update(det_boxes[d_idx])

        # ── Step 4: Spawn new tracks for unmatched detections ─────────────────
        for d_idx in unmatched_dets:
            new_tracker = KalmanBoxTracker(det_boxes[d_idx])
            self.trackers.append(new_tracker)

        # ── Step 5: Remove stale tracks & collect results ─────────────────────
        active_tracks = []
        results       = []

        for tracker in self.trackers:
            if tracker.time_since_update > self.max_age:
                # Track has been missing too long — discard it.
                continue

            active_tracks.append(tracker)

            # Only report tracks that have been confirmed (min_hits matched
            # frames) or are still fresh enough to show (hit_streak > 0).
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = tracker.get_bbox()
                results.append((
                    int(x1), int(y1), int(x2), int(y2),
                    tracker.id
                ))

        self.trackers = active_tracks
        return results
