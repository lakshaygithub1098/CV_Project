import math


class SpeedEstimator:
    """Estimates per-vehicle speed from centroid displacement (pixels/frame)."""

    def __init__(self):
        """Initialize position tracking dict."""
        # Maps track_id -> (cx, cy) from the previous frame.
        self.prev_positions = {}

    def estimate_speed(self, tracks):
        """Compute pixel/frame speed for each vehicle; 0 for new vehicles."""
        speed_dict = {}

        for track in tracks:
            x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], track[4]

            # Centroid of the bounding box.
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if track_id in self.prev_positions:
                prev_x, prev_y = self.prev_positions[track_id]
                distance = math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                speed = distance
            else:
                # First time we see this vehicle — no previous position yet.
                speed = 0

            # Store current position for the next frame.
            self.prev_positions[track_id] = (cx, cy)

            speed_dict[track_id] = speed

        return speed_dict

    def average_speed(self, speed_dict):
        """Return mean speed across all vehicles, or 0 if empty."""
        if not speed_dict:
            return 0

        return sum(speed_dict.values()) / len(speed_dict)
