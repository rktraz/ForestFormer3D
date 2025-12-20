#!/usr/bin/env python3
"""
Test script to check the matching process between prediction and original points.
"""

import os
import sys
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

def test_matching(original_file, prediction_file, tolerance=0.01):
    """Test the matching process."""
    print("=" * 70)
    print("MATCHING TEST")
    print("=" * 70)
    
    # Load original
    print(f"\n1. Loading original file: {original_file}")
    if not os.path.exists(original_file):
        print(f"   ERROR: File not found")
        return False
    
    original_ply = PlyData.read(original_file)
    original_vertex = original_ply["vertex"].data
    orig_x = np.asarray(original_vertex["x"], dtype=np.float64)
    orig_y = np.asarray(original_vertex["y"], dtype=np.float64)
    orig_z = np.asarray(original_vertex["z"], dtype=np.float64)
    orig_points = np.stack([orig_x, orig_y, orig_z], axis=1)
    print(f"   Original points: {len(orig_points):,}")
    
    # Load prediction
    print(f"\n2. Loading prediction file: {prediction_file}")
    if not os.path.exists(prediction_file):
        print(f"   ERROR: File not found")
        return False
    
    pred_ply = PlyData.read(prediction_file)
    pred_vertex = pred_ply["vertex"].data
    pred_x = np.asarray(pred_vertex["x"], dtype=np.float64)
    pred_y = np.asarray(pred_vertex["y"], dtype=np.float64)
    pred_z = np.asarray(pred_vertex["z"], dtype=np.float64)
    pred_points = np.stack([pred_x, pred_y, pred_z], axis=1)
    print(f"   Prediction points: {len(pred_points):,}")
    
    # Check coordinate ranges
    print(f"\n3. Coordinate ranges:")
    print(f"   Original:  x=[{orig_x.min():.2f}, {orig_x.max():.2f}], "
          f"y=[{orig_y.min():.2f}, {orig_y.max():.2f}], "
          f"z=[{orig_z.min():.2f}, {orig_z.max():.2f}]")
    print(f"   Prediction: x=[{pred_x.min():.2f}, {pred_x.max():.2f}], "
          f"y=[{pred_y.min():.2f}, {pred_y.max():.2f}], "
          f"z=[{pred_z.min():.2f}, {pred_z.max():.2f}]")
    
    # Check for offsets
    offset_file = f"data/ForAINetV2/forainetv2_instance_data/107_labeled_offsets.npy"
    offsets = None
    if os.path.exists(offset_file):
        offsets = np.load(offset_file)
        print(f"\n4. Found offsets: {offsets}")
        orig_centered = orig_points.copy()
        orig_centered[:, 0] -= offsets[0]
        orig_centered[:, 1] -= offsets[1]
        orig_centered[:, 2] -= offsets[2]
        print(f"   Centered original: x=[{orig_centered[:, 0].min():.2f}, {orig_centered[:, 0].max():.2f}], "
              f"y=[{orig_centered[:, 1].min():.2f}, {orig_centered[:, 1].max():.2f}], "
              f"z=[{orig_centered[:, 2].min():.2f}, {orig_centered[:, 2].max():.2f}]")
    else:
        print(f"\n4. No offsets file found, using original coordinates")
        orig_centered = orig_points
    
    # Build KDTree and match
    print(f"\n5. Building KDTree and matching (tolerance={tolerance}m)...")
    tree = cKDTree(orig_centered)
    distances, indices = tree.query(pred_points, k=1)
    matched_mask = distances <= tolerance
    num_matched = int(np.sum(matched_mask))
    num_total = len(pred_points)
    
    print(f"   Matched: {num_matched:,}/{num_total:,} ({100*num_matched/num_total:.1f}%)")
    
    if num_matched < num_total:
        unmatched_distances = distances[~matched_mask]
        print(f"   Unmatched points: {num_total - num_matched:,}")
        if len(unmatched_distances) > 0:
            print(f"   Distance stats for unmatched: min={unmatched_distances.min():.4f}m, "
                  f"max={unmatched_distances.max():.4f}m, "
                  f"mean={unmatched_distances.mean():.4f}m, "
                  f"median={np.median(unmatched_distances):.4f}m")
    
    # Test with different tolerances
    print(f"\n6. Testing different tolerances:")
    for tol in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        matched = np.sum(distances <= tol)
        pct = 100 * matched / num_total
        print(f"   Tolerance {tol:5.2f}m: {matched:,}/{num_total:,} ({pct:5.1f}%)")
    
    return num_matched > 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_matching.py <original_file> <prediction_file> [tolerance]")
        sys.exit(1)
    
    original_file = sys.argv[1]
    prediction_file = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    
    success = test_matching(original_file, prediction_file, tolerance)
    sys.exit(0 if success else 1)

