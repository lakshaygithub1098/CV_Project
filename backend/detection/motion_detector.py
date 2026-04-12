import cv2


class MotionDetector:
    """Detects moving vehicles using MOG2 background subtraction."""
    MIN_CONTOUR_AREA = 800  # skip noise
    MAX_CONTOUR_AREA = 30000  # skip merged blobs

    def __init__(self):
        """Setup MOG2 background model."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

    def detect(self, frame):
        """Find moving vehicles and return bboxes + foreground mask."""
        # MOG2 background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Blur and threshold foreground mask
        blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Morphology: remove noise, fill holes, dilate for larger blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours (outlines of moving regions)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,       # only outermost contours
            cv2.CHAIN_APPROX_SIMPLE  # compress straight edges to save memory
        )

        # Filter by area and size; collect bounding boxes
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Too small (noise) or too large (merged vehicles)
            if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Discard too-thin boxes
            if w < 20 or h < 20:
                continue

            # Store as [x1, y1, x2, y2]
            bounding_boxes.append([x, y, x + w, y + h])

        return bounding_boxes, thresh
