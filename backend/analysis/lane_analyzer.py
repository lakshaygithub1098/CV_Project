import cv2
import numpy as np


class LaneAnalyzer:
    """Divides highway into 5 lanes and assigns vehicles to each lane."""

    def __init__(self, roi_x_min=552, roi_x_max=1160, y_top=220, y_bottom=720, taper_pct=0.07):
        """Build 5 lane polygons with perspective taper (narrower at top)."""
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max

        # Perspective taper: road appears slightly narrower at y_top.
        taper     = int((roi_x_max - roi_x_min) * taper_pct)
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
        """Return lane name containing point (x,y), or None."""
        for lane_name, polygon in self.lanes.items():
            # pointPolygonTest returns > 0 if the point is inside the polygon.
            result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
            if result >= 0:
                return lane_name
        return None

    def count_vehicles_per_lane(self, tracks):
        """Count how many vehicles are in each lane (by centroid)."""
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
        """Classify traffic: LIGHT (few+fast), HEAVY (many+slow), else MEDIUM."""
        if total < 6 and avg_speed > 10:
            return "LIGHT"
        elif total > 12 or avg_speed < 4:
            return "HEAVY"
        else:
            return "MEDIUM"

    def draw_lanes(self, frame):
        """Draw all 5 lane polygons on frame in blue."""
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
