import math


class SpeedEstimator:
    """
    Estimates per-vehicle speed by comparing each vehicle's centroid position
    between consecutive frames.

    Speed is expressed in pixels-per-frame.  To convert to real-world units
    (e.g. km/h) you would multiply by a pixel-to-metre scale factor and the
    camera's frame rate — that calibration step is left to the caller.
    """

    def __init__(self):
        # Maps track_id -> (cx, cy) from the previous frame.
        self.prev_positions = {}

    def estimate_speed(self, tracks):
        """
        Compute the pixel-per-frame speed for every tracked vehicle.

        Args:
            tracks (list): List of tracks from SORT.  Each element is a
                           sequence: [x1, y1, x2, y2, track_id].

        Returns:
            dict: {track_id: speed} where speed is a float (pixels/frame).
                  Vehicles seen for the first time in this call have speed 0.
        """
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
        """
        Compute the average speed across all currently tracked vehicles.

        Args:
            speed_dict (dict): Output of estimate_speed(), mapping
                               track_id -> speed.

        Returns:
            float: Mean speed in pixels/frame, or 0 if no vehicles are tracked.
        """
        if not speed_dict:
            return 0

        return sum(speed_dict.values()) / len(speed_dict)
