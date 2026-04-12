"""
Examples of using the Two-Road Analysis System

This file demonstrates:
1. Basic usage of RoadAnalyzer
2. Customizing divider position and colors
3. Accessing road statistics individually
4. Extending for specific use cases
"""

import cv2
from analysis.road_analyzer import RoadAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Basic Usage
# ─────────────────────────────────────────────────────────────────────────────

def example_basic_usage(frame, tracked_vehicles, speeds_dict):
    """Simplest way to use RoadAnalyzer."""
    
    # Initialize once (typically in reset_pipeline)
    road_analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    
    # Compute statistics
    road_stats = road_analyzer.compute_road_statistics(tracked_vehicles, speeds_dict)
    
    # Draw all visualizations
    frame = road_analyzer.draw_road_divider(frame)
    frame = road_analyzer.draw_road_labels(frame)
    frame = road_analyzer.draw_road_statistics(frame, road_stats)
    frame = road_analyzer.draw_detected_vehicles(frame, road_stats, speeds_dict)
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Custom Divider Position
# ─────────────────────────────────────────────────────────────────────────────

def example_custom_divider(frame, tracked_vehicles, speeds_dict):
    """Use a custom divider position (not at center)."""
    
    # Create divider at 70% from left (instead of 50%)
    custom_divider = int(1280 * 0.7)
    
    road_analyzer = RoadAnalyzer(
        frame_width=1280,
        frame_height=720,
        divider_pos=custom_divider
    )
    
    road_stats = road_analyzer.compute_road_statistics(tracked_vehicles, speeds_dict)
    
    # Draw with custom divider position
    frame = road_analyzer.draw_road_divider(frame)
    frame = road_analyzer.draw_road_statistics(frame, road_stats)
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Custom Colors
# ─────────────────────────────────────────────────────────────────────────────

def example_custom_colors(frame, tracked_vehicles, speeds_dict):
    """Use custom colors for roads and visualizations."""
    
    road_analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    road_stats = road_analyzer.compute_road_statistics(tracked_vehicles, speeds_dict)
    
    # Draw with custom colors
    # Road A: Red, Road B: Green
    frame = road_analyzer.draw_road_divider(
        frame,
        color=(0, 0, 255),  # Red in BGR
        thickness=3
    )
    
    frame = road_analyzer.draw_road_statistics(
        frame,
        road_stats,
        road_a_color=(0, 0, 255),      # Red for Road A
        road_b_color=(0, 255, 0),      # Green for Road B
    )
    
    frame = road_analyzer.draw_detected_vehicles(
        frame,
        road_stats,
        speeds_dict,
        road_a_box_color=(0, 0, 255),  # Red boxes for Road A
        road_b_box_color=(0, 255, 0),  # Green boxes for Road B
    )
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Accessing Individual Road Statistics
# ─────────────────────────────────────────────────────────────────────────────

def example_individual_stats(tracked_vehicles, speeds_dict):
    """Access and use individual road statistics."""
    
    road_analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    road_stats = road_analyzer.compute_road_statistics(tracked_vehicles, speeds_dict)
    
    # Access Road A stats
    road_a = road_stats["Road A"]
    print(f"Road A:")
    print(f"  Vehicles: {road_a['vehicle_count']}")
    print(f"  Density: {road_a['density']}")
    print(f"  Avg Speed: {road_a['avg_speed']:.2f} px/fr")
    
    # Access Road B stats
    road_b = road_stats["Road B"]
    print(f"\nRoad B:")
    print(f"  Vehicles: {road_b['vehicle_count']}")
    print(f"  Density: {road_b['density']}")
    print(f"  Avg Speed: {road_b['avg_speed']:.2f} px/fr")
    
    print(f"\nTotal Vehicles: {road_stats['total_vehicles']}")
    
    # Check assignments
    for vehicle_id, road in road_stats['assignments'].items():
        print(f"Vehicle {vehicle_id} → {road}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Traffic Light Control (Use Case)
# ─────────────────────────────────────────────────────────────────────────────

def example_traffic_light_control(road_stats):
    """
    Use road statistics to control traffic lights.
    
    Example logic:
    - If HEAVY congestion on Road A: extend green light for Road A
    - If LIGHT traffic on Road B: shorten green light for Road B
    """
    
    road_a = road_stats["Road A"]
    road_b = road_stats["Road B"]
    
    # Traffic light timing (example)
    if road_a["density"] == "HEAVY":
        road_a_green_time = 45  # seconds
    elif road_a["density"] == "MEDIUM":
        road_a_green_time = 30
    else:
        road_a_green_time = 15
    
    if road_b["density"] == "HEAVY":
        road_b_green_time = 45
    elif road_b["density"] == "MEDIUM":
        road_b_green_time = 30
    else:
        road_b_green_time = 15
    
    print(f"Road A green: {road_a_green_time}s ({road_a['density']})")
    print(f"Road B green: {road_b_green_time}s ({road_b['density']})")
    
    return road_a_green_time, road_b_green_time


# ─────────────────────────────────────────────────────────────────────────────
# Example 6: Speed-Based Alerts
# ─────────────────────────────────────────────────────────────────────────────

def example_speed_alerts(road_stats, speed_limit=20):
    """
    Generate alerts based on average speed per road.
    
    Example use case: Warn if traffic is moving too slowly (potential congestion).
    """
    
    road_a = road_stats["Road A"]
    road_b = road_stats["Road B"]
    
    alerts = []
    
    if road_a["avg_speed"] < speed_limit:
        alerts.append(f"⚠ Road A: Slow traffic ({road_a['avg_speed']:.1f} px/fr)")
    
    if road_b["avg_speed"] < speed_limit:
        alerts.append(f"⚠ Road B: Slow traffic ({road_b['avg_speed']:.1f} px/fr)")
    
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Example 7: Filtering Vehicles by Road
# ─────────────────────────────────────────────────────────────────────────────

def example_filter_by_road(road_stats):
    """Filter vehicles belonging to a specific road."""
    
    road_a_vehicles = road_stats["Road A"]["vehicles"]
    road_b_vehicles = road_stats["Road B"]["vehicles"]
    
    # Process only Road A vehicles
    for x1, y1, x2, y2, vehicle_id in road_a_vehicles:
        print(f"Road A Vehicle {vehicle_id}: bbox=({x1}, {y1}, {x2}, {y2})")
    
    # Process only Road B vehicles
    for x1, y1, x2, y2, vehicle_id in road_b_vehicles:
        print(f"Road B Vehicle {vehicle_id}: bbox=({x1}, {y1}, {x2}, {y2})")


# ─────────────────────────────────────────────────────────────────────────────
# Example 8: Three-Road System (Extension)
# ─────────────────────────────────────────────────────────────────────────────

def example_three_roads(frame, tracked_vehicles, speeds_dict, frame_width=1280):
    """
    Extend the system to support 3 roads instead of 2.
    
    This demonstrates how to customize the system for other configurations.
    """
    
    divider_1 = frame_width // 3
    divider_2 = 2 * frame_width // 3
    
    road_a = []  # x < divider_1
    road_b = []  # divider_1 <= x < divider_2
    road_c = []  # x >= divider_2
    
    for track in tracked_vehicles:
        x1, y1, x2, y2, vehicle_id = track
        cx = (x1 + x2) / 2.0
        
        if cx < divider_1:
            road_a.append(track)
        elif cx < divider_2:
            road_b.append(track)
        else:
            road_c.append(track)
    
    print(f"Road A: {len(road_a)} vehicles")
    print(f"Road B: {len(road_b)} vehicles")
    print(f"Road C: {len(road_c)} vehicles")
    
    # Draw dividers
    cv2.line(frame, (divider_1, 0), (divider_1, frame.shape[0]), (0, 255, 255), 2)
    cv2.line(frame, (divider_2, 0), (divider_2, frame.shape[0]), (0, 255, 255), 2)
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Example 9: Density-Based Heatmap (Advanced Visualization)
# ─────────────────────────────────────────────────────────────────────────────

def example_density_heatmap(frame, road_stats):
    """
    Draw semi-transparent overlays based on traffic density per road.
    
    - LIGHT: Green overlay (low intensity)
    - MEDIUM: Yellow overlay (medium intensity)
    - HEAVY: Red overlay (high intensity)
    """
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    divider = frame_width // 2
    
    # Create overlay
    overlay = frame.copy()
    
    # Road A density overlay (left side)
    road_a_density = road_stats["Road A"]["density"]
    if road_a_density == "LIGHT":
        color = (0, 255, 0)  # Green
        alpha = 0.1
    elif road_a_density == "MEDIUM":
        color = (0, 255, 255)  # Yellow
        alpha = 0.15
    else:  # HEAVY
        color = (0, 0, 255)  # Red
        alpha = 0.2
    
    cv2.rectangle(overlay, (0, 0), (divider, frame_height), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Road B density overlay (right side)
    overlay = frame.copy()
    road_b_density = road_stats["Road B"]["density"]
    if road_b_density == "LIGHT":
        color = (0, 255, 0)
        alpha = 0.1
    elif road_b_density == "MEDIUM":
        color = (0, 255, 255)
        alpha = 0.15
    else:  # HEAVY
        color = (0, 0, 255)
        alpha = 0.2
    
    cv2.rectangle(overlay, (divider, 0), (frame_width, frame_height), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Example 10: Logging and Export (Data Analysis)
# ─────────────────────────────────────────────────────────────────────────────

def example_logging_export(frame_number, road_stats):
    """
    Log road statistics for post-processing and analytics.
    
    Useful for:
    - Generating traffic reports
    - Analyzing congestion patterns
    - Performance metrics
    """
    
    log_data = {
        "frame": frame_number,
        "road_a": {
            "vehicle_count": road_stats["Road A"]["vehicle_count"],
            "density": road_stats["Road A"]["density"],
            "avg_speed": road_stats["Road A"]["avg_speed"],
        },
        "road_b": {
            "vehicle_count": road_stats["Road B"]["vehicle_count"],
            "density": road_stats["Road B"]["density"],
            "avg_speed": road_stats["Road B"]["avg_speed"],
        },
        "total_vehicles": road_stats["total_vehicles"],
    }
    
    print(f"Frame {frame_number}: {log_data}")
    
    # In practice, write to CSV, JSON, or database:
    # import json
    # with open("traffic_log.json", "a") as f:
    #     json.dump(log_data, f)
    #     f.write("\n")
    
    return log_data


# ─────────────────────────────────────────────────────────────────────────────
# Main: Run all examples
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Two-Road Analysis System - Examples")
    print("=" * 50)
    print("\nNote: These are conceptual examples.")
    print("In practice, integrate these into app.py with actual frame data.")
    print("\nRefer to app.py and road_analyzer.py for the actual implementation.")
