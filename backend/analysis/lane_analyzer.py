import cv2
import numpy as np


class LaneAnalyzer:
    """
    Assigns tracked vehicles to one of five perspective-trapezoid lane regions
    on a single-direction highway.

    Lane layout (left → right):
        L1 | L2 | L3 | L4 | L5

    All coordinates are in pixels relative to a 1280×720 frame.
    Each polygon is narrow at the top (far end of road) and wider at the
    bottom (near the camera) to follow the road's vanishing-point perspective.
    Adjust the corner points to match your specific camera angle.
    """

    def __init__(self, roi_x_min=576, roi_x_max=1216, y_top=350, y_bottom=720):
        """
        Build 5 perspective-trapezoid lane polygons that span exactly from
        roi_x_min to roi_x_max at the bottom of the frame, tapering slightly
        inward at y_top to simulate the road's vanishing-point perspective.

        Default values match a 1280×720 frame with the analysis zone covering
        x = 45 %– 95 % of the frame width (the main highway lanes).

        Args:
            roi_x_min (int): Left  edge of the analysis zone in pixels.
            roi_x_max (int): Right edge of the analysis zone in pixels.
            y_top     (int): Y coordinate of the far (top)  lane boundary.
            y_bottom  (int): Y coordinate of the near (bottom) lane boundary.
        """
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max

        # Perspective taper: road appears ~5 % narrower on each side at y_top.
        taper     = int((roi_x_max - roi_x_min) * 0.05)
        top_left  = roi_x_min + taper
        top_right = roi_x_max - taper
        bot_left  = roi_x_min
        bot_right = roi_x_max

        top_width = top_right - top_left
        bot_width = bot_right - bot_left
        n = 5

        lane_names = ["L1", "L2", "L3", "L4", "L5"]
        self.lanes = {}
        for i, name in enumerate(lane_names):
            tx_l = int(top_left + i       * top_width / n)
            tx_r = int(top_left + (i + 1) * top_width / n)
            bx_l = int(bot_left + i       * bot_width / n)
            bx_r = int(bot_left + (i + 1) * bot_width / n)
            self.lanes[name] = np.array([
                [tx_l, y_top], [tx_r, y_top],
                [bx_r, y_bottom], [bx_l, y_bottom]
            ])

    def get_lane_for_point(self, x, y):
        """
        Return the name of the lane that contains the point (x, y), or
        None if the point is not inside any lane polygon.

        Args:
            x (int): Horizontal coordinate.
            y (int): Vertical coordinate.

        Returns:
            str | None: Lane name e.g. "A1", "B3", or None.
        """
        for lane_name, polygon in self.lanes.items():
            # pointPolygonTest returns > 0 if the point is inside the polygon.
            result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
            if result >= 0:
                return lane_name
        return None

    def count_vehicles_per_lane(self, tracks):
        """
        Count how many tracked vehicles are in each lane.

        Args:
            tracks (list): List of tracks from SORT, each as
                           (x1, y1, x2, y2, track_id).

        Returns:
            dict: {"L1": count, "L2": count, "L3": count, "L4": count, "L5": count}
        """
        counts = {lane: 0 for lane in self.lanes}

        for track in tracks:
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]

            # Centroid of the bounding box.
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            lane = self.get_lane_for_point(cx, cy)
            if lane is not None:
                counts[lane] += 1

        return counts

    def compute_congestion(self, total, avg_speed=0):
        """
        Return a congestion level based on vehicle count AND average speed.

        Using both metrics gives a more realistic picture: a road with few
        vehicles moving slowly is still congested, and a fast-moving road
        with many vehicles may still be flowing freely.

        Args:
            total     (int):   Total number of vehicles currently tracked.
            avg_speed (float): Average speed in pixels/frame from SpeedEstimator.
                               Defaults to 0 when speed data is unavailable.

        Returns:
            str: "LIGHT", "MEDIUM", or "HEAVY".
        """
        if total < 6 and avg_speed > 10:
            return "LIGHT"
        elif total > 12 or avg_speed < 4:
            return "HEAVY"
        else:
            return "MEDIUM"

    def draw_lanes(self, frame):
        """
        Draw all lane polygons on the frame for visual debugging.

        All highway lanes are drawn in blue (255, 0, 0).

        Args:
            frame (numpy.ndarray): The BGR video frame to draw on (modified
                                   in-place).

        Returns:
            numpy.ndarray: The same frame with lane outlines drawn.
        """
        color = (255, 0, 0)   # blue for all highway lanes

        for lane_name, polygon in self.lanes.items():
            cv2.polylines(frame, [polygon], isClosed=True, color=color,
                          thickness=2)

            # Label each lane at its centroid.
            cx = int(polygon[:, 0].mean())
            cy = int(polygon[:, 1].mean())
            cv2.putText(frame, lane_name, (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame
