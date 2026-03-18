import cv2
import glob
import os
import sys

# ── Fix display issues on Linux / GNOME / Wayland ─────────────────────────────
# OpenCV's Qt highgui must be told to use the X11/XCB backend.
# On Wayland desktops (Ubuntu 22+, Fedora 36+, etc.) it otherwise either
# fails to open a window or floods the terminal with font warnings.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_FONT_DPI", "96")

# Make sure Python can find the backend packages regardless of where the
# script is launched from.
sys.path.insert(0, os.path.dirname(__file__))

from detection.motion_detector  import MotionDetector
from tracking.sort_tracker      import SortTracker
from analysis.lane_analyzer     import LaneAnalyzer
from analysis.speed_estimator   import SpeedEstimator


# ── Configuration ─────────────────────────────────────────────────────────────

# Directory (relative to project root) that contains the video files.
VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "videos")

# Colour and thickness of the bounding-box rectangles (BGR format).
BOX_COLOR     = (0, 255, 0)   # green
BOX_THICKNESS = 2

# SORT tracker parameters.
MAX_AGE       = 5
MIN_HITS      = 1
IOU_THRESHOLD = 0.3

# ROI (Region of Interest) for lane analysis.
# Frames are always resized to 1280×720, so these pixel values are constant.
# Vehicles whose centroid falls between ROI_X_MIN and ROI_X_MAX are counted
# for lane assignment, speed, and congestion.  Vehicles outside the ROI are
# still detected and drawn but do not affect the traffic statistics.
FRAME_W   = 1280
ROI_X_MIN = int(FRAME_W * 0.45)   # 576  — left  edge of highway lanes
ROI_X_MAX = int(FRAME_W * 0.95)   # 1216 — right edge of highway lanes


# ── Per-video processing ──────────────────────────────────────────────────────

def process_video(video_path, analyzer):
    """
    Open a single video file, run the full pipeline on every frame, and
    display the results in two windows until the video ends or ESC is pressed.

    A fresh MotionDetector and SortTracker are created for each video so that
    the MOG2 background model and track IDs always start clean.

    Args:
        video_path (str): Absolute or relative path to the video file.
        analyzer   (LaneAnalyzer): Pre-built lane analyser (shared across videos).

    Returns:
        bool: True if the video finished normally, False if ESC was pressed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path} — skipping.")
        return True

    print(f"\n[INFO] Processing: {os.path.basename(video_path)}")
    print("[INFO] Press ESC to skip to the next video.")

    # Create a fresh detector, tracker, and speed estimator for this video.
    # Each video starts with a clean background model, track IDs, and
    # position history so readings don't bleed between files.
    detector  = MotionDetector()
    tracker   = SortTracker(
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD,
    )
    estimator = SpeedEstimator()

    # Create (or re-use) the display windows.
    cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Detection", 1280, 720)
    cv2.namedWindow("Foreground Mask",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foreground Mask",   640, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video.
            break

        # ── 1. Resize to 1280×720 for consistent processing ───────────────────
        frame = cv2.resize(frame, (1280, 720))

        # ── 2. Detect moving vehicles ─────────────────────────────────────────
        bounding_boxes, fg_mask = detector.detect(frame)

        # ── 3. Track vehicles across frames with SORT ─────────────────────────
        # Append score=1.0 to each detection before passing to the tracker.
        detections_for_tracker = [
            [x1, y1, x2, y2, 1.0]
            for (x1, y1, x2, y2) in bounding_boxes
        ]
        tracked_vehicles = tracker.update(detections_for_tracker)

        # ── 4. ROI filter — only vehicles on the main highway (45 %–95 % of frame) ─
        # Detection and drawing use all tracked_vehicles.
        # Lane counting, speed, and congestion use only lane_tracks so that
        # vehicles on adjacent roads or at frame edges don't skew results.
        roi_x_min = ROI_X_MIN
        roi_x_max = ROI_X_MAX
        lane_tracks = [
            t for t in tracked_vehicles
            if roi_x_min < int((t[0] + t[2]) / 2) < roi_x_max
        ]

        # ── 5. Speed estimation (must run before congestion) ─────────────────
        speeds    = estimator.estimate_speed(lane_tracks)
        avg_speed = estimator.average_speed(speeds)

        # ── 6. Lane analysis and congestion (ROI vehicles only) ───────────────
        lane_counts = analyzer.count_vehicles_per_lane(lane_tracks)
        congestion  = analyzer.compute_congestion(sum(lane_counts.values()), avg_speed)

        # ── 7. Draw ROI boundary lines ──────────────────────────────────────────
        cv2.line(frame, (roi_x_min, 0), (roi_x_min, frame.shape[0]),
                 (0, 255, 255), 2)
        cv2.line(frame, (roi_x_max, 0), (roi_x_max, frame.shape[0]),
                 (0, 255, 255), 2)
        cv2.putText(frame, "Analysis Zone", (roi_x_min + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # ── 8. Draw lane outlines (so boxes appear on top) ────────────────────
        analyzer.draw_lanes(frame)

        # ── 9. Draw bounding boxes, IDs, and per-vehicle speed ────────────────
        # Use all tracked_vehicles so detections outside the ROI are still shown.
        for (x1, y1, x2, y2, vehicle_id) in tracked_vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            spd     = speeds.get(vehicle_id, 0)
            label   = f"ID:{int(vehicle_id)}  {spd:.0f}px"
            label_y = max(y1 - 5, 15)   # keep label inside frame
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1, cv2.LINE_AA)

        # ── 7. Overlay stats panel (top-left) ─────────────────────────────────
        congestion_color = (0, 255, 0)     # green  – LIGHT
        if congestion == "MEDIUM":
            congestion_color = (0, 165, 255)   # orange
        elif congestion == "HEAVY":
            congestion_color = (0, 0, 255)     # red

        overlay_lines = [
            (f"Vehicles: {len(tracked_vehicles)}",      (0, 255, 0)),
            (f"L1: {lane_counts['L1']}",                (255, 200, 0)),
            (f"L2: {lane_counts['L2']}",                (255, 200, 0)),
            (f"L3: {lane_counts['L3']}",                (255, 200, 0)),
            (f"L4: {lane_counts['L4']}",                (255, 200, 0)),
            (f"L5: {lane_counts['L5']}",                (255, 200, 0)),
            (f"Traffic: {congestion}",                  congestion_color),
            (f"Avg Speed: {avg_speed:.1f} px/fr",       (0, 220, 255)),
        ]
        for i, (text, color) in enumerate(overlay_lines):
            cv2.putText(frame, text, (20, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Current filename at the bottom of the frame so it's always visible.
        cv2.putText(frame, os.path.basename(video_path),
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 8. Show windows ───────────────────────────────────────────────────
        cv2.imshow("Traffic Detection", frame)
        fg_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Foreground Mask", fg_display)

        # ESC skips to the next video; any other key continues.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] ESC pressed — skipping to next video.")
            cap.release()
            return False   # signal: user pressed ESC

    cap.release()
    return True   # signal: video finished normally


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Collect all supported video files in the videos directory.
    videos_dir  = os.path.normpath(VIDEOS_DIR)
    video_files = (
        glob.glob(os.path.join(videos_dir, "*.avi")) +
        glob.glob(os.path.join(videos_dir, "*.mp4")) +
        glob.glob(os.path.join(videos_dir, "*.mov")) +
        glob.glob(os.path.join(videos_dir, "*.MOV"))
    )
    video_files.sort()   # process in alphabetical order

    if not video_files:
        print(f"[ERROR] No video files found in: {videos_dir}")
        print("Supported formats: .avi  .mp4  .mov")
        sys.exit(1)

    print(f"[INFO] Found {len(video_files)} video(s) in {videos_dir}")

    # The lane analyser is built once and reused across all videos because
    # the lane geometry stays the same for a fixed camera.
    analyzer = LaneAnalyzer(roi_x_min=ROI_X_MIN, roi_x_max=ROI_X_MAX)

    for idx, video_path in enumerate(video_files, start=1):
        print(f"[INFO] Video {idx}/{len(video_files)}: {os.path.basename(video_path)}")
        process_video(video_path, analyzer)

    cv2.destroyAllWindows()
    print("[INFO] All videos processed. Exiting.")


if __name__ == "__main__":
    main()

