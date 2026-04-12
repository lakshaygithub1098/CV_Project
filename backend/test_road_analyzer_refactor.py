#!/usr/bin/env python3
"""
Road Analyzer Refactoring - Validation & Testing Script

Tests the new RoadAnalyzer features:
1. Basic centroid-based assignment
2. Anti-flickering buffer zone
3. Lane distribution  
4. Density classification
5. Integration with existing pipeline

Usage:
    python backend/test_road_analyzer_refactor.py
"""

import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from analysis.road_analyzer import RoadAnalyzer


def test_basic_assignment():
    """Test basic vehicle assignment to roads."""
    print("\n" + "="*60)
    print("TEST 1: Basic Vehicle Assignment")
    print("="*60)
    
    analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    
    # Test vehicles
    tracks = [
        [100, 100, 200, 200, 1],    # Left side, Road A
        [600, 100, 700, 200, 2],    # Center area, Road B (centroid at 650)
        [50, 50, 150, 150, 3],      # Left, Road A
        [1100, 100, 1200, 200, 4],  # Right, Road B
    ]
    
    result = analyzer.assign_vehicles_to_roads(tracks)
    
    print(f"\n✓ Road A vehicles: {len(result['Road A'])} (expected: 2)")
    print(f"  IDs: {[v[4] for v in result['Road A']]}")
    
    print(f"\n✓ Road B vehicles: {len(result['Road B'])} (expected: 2)")
    print(f"  IDs: {[v[4] for v in result['Road B']]}")
    
    print(f"\n✓ Assignments: {result['assignments']}")
    
    # Validate
    assert len(result['Road A']) == 2, "Road A should have 2 vehicles"
    assert len(result['Road B']) == 2, "Road B should have 2 vehicles"
    assert result['assignments'][1] == "Road A"
    assert result['assignments'][2] == "Road B"
    
    print("\n✅ TEST 1 PASSED: Assignment working correctly")


def test_centroid_calculation():
    """Test centroid-based assignment (not bounding box overlap)."""
    print("\n" + "="*60)
    print("TEST 2: Centroid-Based Assignment")
    print("="*60)
    
    analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    
    # Vehicle that overlaps divider but centroid is in Road B
    # BBox: x1=620, x2=660 → centroid=640 (exactly at center)
    # But x2=680 would give centroid=650 (in Road B)
    track = [620, 100, 680, 200, 1]  # centroid = (620+680)/2 = 650
    
    cx, cy = analyzer.get_centroid(track)
    road = analyzer.get_road_for_centroid(cx)
    
    print(f"\nBounding box: x1=620, x2=680")
    print(f"Centroid: x={cx}, y={cy}")
    print(f"✓ Centroid > 640 (divider), assigned to: {road}")
    
    assert cx == 650, f"Centroid should be 650, got {cx}"
    assert road == "Road B", f"Should be Road B, got {road}"
    
    print("\n✅ TEST 2 PASSED: Centroid-based assignment works")


def test_buffer_zone():
    """Test anti-flickering buffer zone."""
    print("\n" + "="*60)
    print("TEST 3: Anti-Flickering Buffer Zone")
    print("="*60)
    
    analyzer = RoadAnalyzer(frame_width=1280, frame_height=720, buffer_zone=50)
    
    # First frame: vehicle slightly left of divider (Road A)
    track1 = [600, 100, 660, 200, 1]  # cx = 630
    road1 = analyzer.assign_vehicle_to_road(track1)
    print(f"\nFrame 1: Vehicle at x=630 (left of divider)")
    print(f"  → Assigned to: {road1}")
    
    # Second frame: vehicle crosses into buffer zone (Road B territory)
    # but buffer zone should keep it in Road A
    track2 = [640, 100, 700, 200, 1]  # cx = 670 (in buffer zone from divider)
    road2 = analyzer.assign_vehicle_to_road(track2)
    print(f"\nFrame 2: Vehicle at x=670 (within 50px buffer)")
    print(f"  → Still assigned to: {road2} (stuck in buffer zone)")
    
    # Third frame: vehicle far beyond buffer zone (Road B)
    track3 = [750, 100, 850, 200, 1]  # cx = 800 (far right)
    road3 = analyzer.assign_vehicle_to_road(track3)
    print(f"\nFrame 3: Vehicle at x=800 (far from divider)")
    print(f"  → Now assigned to: {road3} (escaped buffer zone)")
    
    assert road1 == "Road A", "First assignment should be Road A"
    assert road2 == "Road A", "Buffer zone should keep it in Road A"
    assert road3 == "Road B", "Far right should switch to Road B"
    
    print("\n✅ TEST 3 PASSED: Buffer zone prevents flickering")


def test_density_classification():
    """Test traffic density classification."""
    print("\n" + "="*60)
    print("TEST 4: Density Classification")
    print("="*60)
    
    analyzer = RoadAnalyzer()
    
    test_cases = [
        (0, "LIGHT"),
        (3, "LIGHT"),
        (5, "LIGHT"),
        (6, "MEDIUM"),
        (8, "MEDIUM"),
        (10, "MEDIUM"),
        (11, "HEAVY"),
        (15, "HEAVY"),
        (100, "HEAVY"),
    ]
    
    print("\nVehicle Count → Density Classification:")
    for count, expected in test_cases:
        result = analyzer.calculate_density_category(count)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {count:3} vehicles → {result:6} (expected: {expected})")
        assert result == expected, f"Count {count} should be {expected}, got {result}"
    
    print("\n✅ TEST 4 PASSED: Density classification correct")


def test_statistics_computation():
    """Test full statistics computation."""
    print("\n" + "="*60)
    print("TEST 5: Statistics Computation")
    print("="*60)
    
    analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    
    # Create test tracks
    tracks = [
        # Road A vehicles
        [100, 100, 200, 200, 1],
        [250, 100, 350, 200, 2],
        [400, 100, 500, 200, 3],
        # Road B vehicles
        [700, 100, 800, 200, 4],
        [900, 100, 1000, 200, 5],
    ]
    
    # Create speed dictionary
    speeds = {
        1: 40.0,
        2: 45.0,
        3: 50.0,
        4: 55.0,
        5: 60.0,
    }
    
    # Compute stats
    road_stats = analyzer.compute_road_stats(tracks, speeds)
    
    print(f"\n✓ Total vehicles: {road_stats['total']}")
    print(f"\n✓ Road A Statistics:")
    print(f"    - Vehicle count: {road_stats['Road A']['count']}")
    print(f"    - Density: {road_stats['Road A']['density']}")
    print(f"    - Avg speed: {road_stats['Road A']['avg_speed']:.2f} px/fr")
    
    print(f"\n✓ Road B Statistics:")
    print(f"    - Vehicle count: {road_stats['Road B']['count']}")
    print(f"    - Density: {road_stats['Road B']['density']}")
    print(f"    - Avg speed: {road_stats['Road B']['avg_speed']:.2f} px/fr")
    
    # Validate
    assert road_stats['Road A']['count'] == 3
    assert road_stats['Road B']['count'] == 2
    assert road_stats['total'] == 5
    assert abs(road_stats['Road A']['avg_speed'] - 45.0) < 0.01  # (40+45+50)/3
    assert abs(road_stats['Road B']['avg_speed'] - 57.5) < 0.01  # (55+60)/2
    
    print("\n✅ TEST 5 PASSED: Statistics computation correct")


def test_lane_distribution():
    """Test lane-wise vehicle distribution."""
    print("\n" + "="*60)
    print("TEST 6: Lane Distribution")
    print("="*60)
    
    analyzer = RoadAnalyzer(
        frame_width=1280,
        frame_height=720,
        enable_lanes=True,
        num_lanes=5
    )
    
    # Create 5 vehicles spread across Road A lanes
    # Left side is 0-640, divided into 5 lanes: 0-128, 128-256, 256-384, 384-512, 512-640
    tracks = [
        [10, 100, 50, 200, 1],      # Lane 0 (0-128)
        [150, 100, 200, 200, 2],    # Lane 1 (128-256)
        [300, 100, 350, 200, 3],    # Lane 2 (256-384)
        [450, 100, 500, 200, 4],    # Lane 3 (384-512)
        [580, 100, 620, 200, 5],    # Lane 4 (512-640)
    ]
    
    speeds = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    road_stats = analyzer.compute_road_stats(tracks, speeds)
    
    print(f"\n✓ Road A Lane Distribution:")
    for lane in road_stats['Road A']['lanes']:
        print(f"    Lane {lane['index']}: {lane['count']} vehicles → IDs: {lane['vehicles']}")
    
    # Validate
    assert road_stats['Road A']['count'] == 5
    for i, lane in enumerate(road_stats['Road A']['lanes']):
        assert lane['count'] == 1, f"Lane {i} should have 1 vehicle"
        assert len(lane['vehicles']) == 1
    
    print("\n✅ TEST 6 PASSED: Lane distribution works")


def test_backwards_compatibility():
    """Test that old API still works."""
    print("\n" + "="*60)
    print("TEST 7: Backwards Compatibility")
    print("="*60)
    
    analyzer = RoadAnalyzer(frame_width=1280, frame_height=720)
    
    tracks = [
        [100, 100, 200, 200, 1],
        [700, 100, 800, 200, 2],
    ]
    speeds = {1: 40.0, 2: 50.0}
    
    # Old API call
    road_stats_old = analyzer.compute_road_statistics(tracks, speeds)
    
    print(f"\n✓ Old API (compute_road_statistics) works:")
    print(f"    Road A count: {road_stats_old['Road A']['vehicle_count']}")
    print(f"    Road B count: {road_stats_old['Road B']['vehicle_count']}")
    print(f"    Total: {road_stats_old['total_vehicles']}")
    
    # New API call
    road_stats_new = analyzer.compute_road_stats(tracks, speeds)
    
    print(f"\n✓ New API (compute_road_stats) works:")
    print(f"    Road A count: {road_stats_new['Road A']['count']}")
    print(f"    Road B count: {road_stats_new['Road B']['count']}")
    print(f"    Total: {road_stats_new['total']}")
    
    # Validate both work
    assert road_stats_old['Road A']['vehicle_count'] == 1
    assert road_stats_new['Road A']['count'] == 1
    
    print("\n✅ TEST 7 PASSED: Backwards compatibility maintained")


def test_color_scheme():
    """Test density-based color scheme."""
    print("\n" + "="*60)
    print("TEST 8: Color Scheme")
    print("="*60)
    
    analyzer = RoadAnalyzer()
    
    test_cases = [
        ("LIGHT", (0, 255, 0), "GREEN"),
        ("MEDIUM", (0, 165, 255), "ORANGE"),
        ("HEAVY", (0, 0, 255), "RED"),
    ]
    
    print("\nDensity → Color (BGR):")
    for density, expected_color, color_name in test_cases:
        color = analyzer.get_density_color(density)
        status = "✓" if color == expected_color else "✗"
        print(f"  {status} {density:6} → {color} ({color_name})")
        assert color == expected_color
    
    print("\n✅ TEST 8 PASSED: Color scheme correct")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print(" ROAD ANALYZER REFACTORING - VALIDATION TEST SUITE")
    print("="*80)
    
    tests = [
        test_basic_assignment,
        test_centroid_calculation,
        test_buffer_zone,
        test_density_classification,
        test_statistics_computation,
        test_lane_distribution,
        test_backwards_compatibility,
        test_color_scheme,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f" SUMMARY: {passed} PASSED, {failed} FAILED")
    print("="*80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Road Analyzer Refactoring is Production-Ready!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
