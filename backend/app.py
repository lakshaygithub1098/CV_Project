import cv2
import os
import sys

# ── Fix display issues on Linux / GNOME / Wayland ─────────────────────────────
# OpenCV's Qt highgui must be told to use the X11/XCB backend.
# On Wayland desktops (Ubuntu 22+, Fedora 36+, etc.) it otherwise either
# fails to open a window or floods the terminal with font warnings.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_FONT_DPI", "96")

# Make sure Python can find the detection package regardless of where the
# script is launched from.
sys.path.insert(0, os.path.dirname(__file__))

from detection.motion_detector import MotionDetector
from tracking.sort_tracker    import SortTracker


# ── Configuration ──────────────────────────────────────────────────────────────

# Path to the input video, relative to this file's location.
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "videos", "input-001.MOV")

# Minimum contour area (in pixels) to be counted as a vehicle.
# 1500 px² is a good balance — filters noise while catching cars/motorcycles.
# Raise if you get too many false positives; lower for distant/small vehicles.
MIN_CONTOUR_AREA = 1500

# Colour and thickness of the bounding-box rectangles (BGR format).
BOX_COLOR     = (0, 255, 0)   # green
BOX_THICKNESS = 2

# SORT tracker parameters.
# max_age      – frames a track survives without a matching detection.
# min_hits     – detections needed before a track is reported (1 = show immediately).
# iou_threshold – minimum IoU to link a detection to an existing track.
MAX_AGE       = 5
MIN_HITS      = 1
IOU_THRESHOLD = 0.3


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Open the video file ─────────────────────────────────────────────────
    video_path = os.path.normpath(VIDEO_PATH)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        print("Make sure the file exists at  data/videos/input-001.MOV")
        sys.exit(1)

    print(f"[INFO] Loaded video: {video_path}")
    print("[INFO] Press ESC to quit.")

    # ── 2. Create named display windows before the loop ───────────────────────
    # WINDOW_NORMAL lets the user resize windows freely.
    # resizeWindow sets the initial size so the video fits on most screens.
    cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Detection", 1280, 720)
    cv2.namedWindow("Foreground Mask",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foreground Mask",   640, 360)

    # ── 3. Create the detector and the SORT tracker ──────────────────────────
    detector = MotionDetector()
    tracker  = SortTracker(
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD
    )

    # ── 4. Process frames in a loop ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()

        # ret is False when the video ends or a read error occurs.
        if not ret:
            print("[INFO] End of video or cannot read frame. Exiting.")
            break

        # Resize every frame to 1280×720 before processing.
        # This keeps coordinates consistent and ensures the window fits on screen.
        frame = cv2.resize(frame, (1280, 720))

        # ── 5. Detect moving vehicles ─────────────────────────────────────────
        # detector.detect() returns (x, y, w, h) boxes and the foreground mask.
        bounding_boxes, fg_mask = detector.detect(frame)

        # ── 6. Track vehicles across frames with SORT ─────────────────────────
        # Detector now returns [x1, y1, x2, y2]; append score=1.0 for SORT.
        detections_for_tracker = [
            [x1, y1, x2, y2, 1.0]
            for (x1, y1, x2, y2) in bounding_boxes
        ]
        tracked_vehicles = tracker.update(detections_for_tracker)

        # ── 7. Draw bounding boxes and vehicle IDs ────────────────────────────
        for (x1, y1, x2, y2, vehicle_id) in tracked_vehicles:
            # Bounding box
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS
            )

            # ID label — drawn 5 px above the top-left corner of the box.
            label      = f"ID: {vehicle_id}"
            label_y    = max(y1 - 5, 15)   # clamp so label stays inside frame
            cv2.putText(
                frame, label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BOX_COLOR,
                2,
                cv2.LINE_AA
            )

        # Vehicle count banner at the top of the frame
        count_text = f"Vehicles tracked: {len(tracked_vehicles)}"
        cv2.putText(
            frame, count_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
            cv2.LINE_AA
        )

        # ── 8. Display the two windows ────────────────────────────────────────
        cv2.imshow("Traffic Detection", frame)

        # Convert the single-channel mask to 3-channel so it renders as a
        # proper greyscale image on all platforms.
        fg_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Foreground Mask", fg_display)

        # ── 9. Wait for a key press; ESC (key code 27) exits ─────────────────
        # waitKey(1) keeps the loop as fast as possible while still allowing
        # OpenCV to process window events and key presses.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            print("[INFO] ESC pressed. Exiting.")
            break

    # ── 10. Clean up ────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
