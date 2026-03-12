# AI Traffic Manager

A computer vision project that detects and tracks vehicles in traffic footage using Python and OpenCV.

## How it works

```
Video → MOG2 Background Subtraction → Contour Detection → SORT Tracking → Display
```

1. **Background subtraction** (MOG2) separates moving vehicles from the static road background
2. **Contour detection** finds the outlines of each moving blob and filters noise by area
3. **SORT tracker** assigns a persistent ID to each vehicle across frames using a Kalman Filter and the Hungarian algorithm
4. **Display** draws bounding boxes and IDs on the live video feed

## Project structure

```
TrafficManagerCV/
├── backend/
│   ├── app.py                    # Entry point — loads video and runs the pipeline
│   ├── detection/
│   │   └── motion_detector.py    # MOG2 background subtraction + contour filtering
│   └── tracking/
│       └── sort_tracker.py       # SORT: Kalman Filter + Hungarian matching
└── data/
    └── videos/
        └── input-001.MOV         # Input traffic video (not included in repo)
```

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- SciPy

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install opencv-python numpy scipy
```

## Run

```bash
cd backend
python app.py
```

Press **ESC** to quit.

## Features

- MOG2 background subtraction with morphological mask cleaning
- Contour-based vehicle detection with area filtering (800 – 30 000 px²)
- SORT multi-object tracking with persistent vehicle IDs
- Two display windows: **Traffic Detection** and **Foreground Mask**
- Live vehicle count overlay
