#!/usr/bin/env python3
"""
Test script to check standardization and point counts at each step.
"""

import os
import sys
from pathlib import Path
from plyfile import PlyData
import numpy as np

def count_points(file_path):
    """Count points in a PLY file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    try:
        ply = PlyData.read(file_path)
        return len(ply["vertex"]), None
    except Exception as e:
        return None, str(e)

def check_file_fields(file_path):
    """Check what fields are in a PLY file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    try:
        ply = PlyData.read(file_path)
        vertex = ply["vertex"]
        fields = list(vertex.dtype.names)
        return fields, None
    except Exception as e:
        return None, str(e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_standardization.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("STANDARDIZATION TEST")
    print("=" * 70)
    
    # 1. Check original file
    print(f"\n1. Original file: {input_file}")
    num_points, error = count_points(input_file)
    if error:
        print(f"   ERROR: {error}")
        sys.exit(1)
    print(f"   Points: {num_points:,}")
    
    fields, error = check_file_fields(input_file)
    if error:
        print(f"   ERROR reading fields: {error}")
    else:
        print(f"   Fields: {', '.join(fields[:10])}{'...' if len(fields) > 10 else ''}")
    
    # 2. Standardize file
    print(f"\n2. Standardizing file...")
    try:
        from tools.standardize_with_cache import get_standardized_file
        standardized_file = get_standardized_file(
            input_file,
            preserve_labels=False,
            force_recompute=False
        )
        
        if standardized_file is None:
            print("   ERROR: Standardization failed")
            sys.exit(1)
        
        print(f"   Standardized file: {standardized_file}")
        num_points_std, error = count_points(standardized_file)
        if error:
            print(f"   ERROR: {error}")
        else:
            print(f"   Points: {num_points_std:,}")
            if num_points_std != num_points:
                print(f"   ⚠️  WARNING: Point count changed! ({num_points:,} -> {num_points_std:,})")
            else:
                print(f"   ✅ Point count preserved")
        
        fields_std, error = check_file_fields(standardized_file)
        if error:
            print(f"   ERROR reading fields: {error}")
        else:
            print(f"   Fields: {', '.join(fields_std[:10])}{'...' if len(fields_std) > 10 else ''}")
    
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 3. Check what would be copied to test_data
    print(f"\n3. File that would be used for inference:")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    test_data_file = f"data/ForAINetV2/test_data/{base_name}.ply"
    if os.path.exists(test_data_file):
        num_points_test, error = count_points(test_data_file)
        if error:
            print(f"   ERROR: {error}")
        else:
            print(f"   File: {test_data_file}")
            print(f"   Points: {num_points_test:,}")
    else:
        print(f"   File doesn't exist yet (will be created during inference)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original:     {num_points:,} points")
    if standardized_file:
        print(f"Standardized: {num_points_std:,} points")
        if num_points_std == num_points:
            print("✅ Standardization preserves all points")
        else:
            print(f"⚠️  Standardization lost {num_points - num_points_std:,} points")
    
    print("\nNote: During inference, the model processes a SUBSET of points")
    print("      (after CylinderCrop, GridSample, PointSample).")
    print("      The final output only contains points that were processed by the model.")

if __name__ == "__main__":
    main()

