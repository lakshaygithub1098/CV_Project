import cv2


class MotionDetector:
    """
    Detects moving vehicles in a video using MOG2 background subtraction.

    MOG2 (Mixture of Gaussians) builds a model of the background over time,
    so anything that moves away from that background is highlighted as foreground.
    """

    # Area thresholds (pixels²) for contour filtering.
    # MIN – blobs smaller than this are noise (shadows, leaves, etc.).
    # MAX – blobs larger than this are merged vehicles; reject them so one
    #        giant box doesn't swallow an entire lane of cars.
    MIN_CONTOUR_AREA = 800
    MAX_CONTOUR_AREA = 30000

    def __init__(self):
        """
        Initialize the MotionDetector.
        """
        # Create the MOG2 background subtractor.
        # history      – how many past frames are used to build the background model.
        # varThreshold – sensitivity; higher value = less sensitive to subtle changes.
        # detectShadows – set to False so shadows don't create extra blobs.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

    def detect(self, frame):
        """
        Run detection on a single video frame.

        Args:
            frame (numpy.ndarray): A BGR image from cv2.VideoCapture.

        Returns:
            bounding_boxes (list of lists): Each entry is [x1, y1, x2, y2].
            fg_mask (numpy.ndarray):        The cleaned foreground mask (grayscale).
        """
        # ── Step 1: Apply background subtraction ──────────────────────────────
        # White pixels in fg_mask = moving foreground; black = background.
        fg_mask = self.bg_subtractor.apply(frame)

        # ── Step 2: Clean up the mask ──────────────────────────────────────────
        # Gaussian blur softens noise before thresholding.
        blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # Strict black-and-white mask for contour detection.
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # OPEN first removes tiny specks; CLOSE fills holes inside blobs.
        # 2 dilation passes expand blobs so nearby fragments of the same
        # vehicle merge into one solid region.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # ── Step 3: Find contours ──────────────────────────────────────────────
        # Each contour is an outline around one connected white region.
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,       # only outermost contours
            cv2.CHAIN_APPROX_SIMPLE  # compress straight edges to save memory
        )

        # ── Step 4: Filter contours and collect bounding boxes ─────────────────
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Too small — noise, shadow flicker, or a leaf.
            if area < self.MIN_CONTOUR_AREA:
                continue

            # Too large — multiple vehicles merged into one blob.
            if area > self.MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Discard flat/thin boxes that are clearly not vehicles.
            if w < 20 or h < 20:
                continue

            # Store as [x1, y1, x2, y2] (top-left and bottom-right corners).
            bounding_boxes.append([x, y, x + w, y + h])

        return bounding_boxes, thresh
