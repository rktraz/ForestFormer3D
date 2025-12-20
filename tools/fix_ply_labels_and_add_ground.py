#!/usr/bin/env python3
"""
Fix PLY files by correcting semantic label mapping and adding dummy ground points.

This script:
1. Fixes semantic label mapping: 2,3 → 1,2 (or keeps 1,2 if already correct)
2. Adds dummy ground points (class 0) at the bottom of each point cloud
3. Ensures proper treeID assignment

Usage:
    python tools/fix_ply_labels_and_add_ground.py --input-dir data/ForAINetV2/train_val_data
    python tools/fix_ply_labels_and_add_ground.py --input-dir /path/to/ply/files --output-dir /path/to/output
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def read_ply_file(ply_path):
    """Read PLY file and return data dictionary."""
    try:
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        
        # Convert to dictionary for easier access
        data = {}
        for name in vertex_data.dtype.names:
            data[name] = np.asarray(vertex_data[name])
        
        return data, len(vertex_data)
    except Exception as e:
        raise RuntimeError(f"Failed to read PLY file {ply_path}: {e}")


def fix_ply_file(input_path, output_path=None, num_ground_points=1000, ground_height_offset=-0.5):
    """
    Fix a single PLY file by correcting labels and adding ground points.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file (if None, overwrites input)
        num_ground_points: Number of dummy ground points to add
        ground_height_offset: Z offset for ground points (relative to min Z)
    
    Returns:
        tuple: (success: bool, num_points: int, num_ground_added: int, message: str)
    """
    if output_path is None:
        output_path = input_path
    
    try:
        # Step 1: Read PLY file
        data, num_points = read_ply_file(input_path)
        
        # Step 2: Extract required fields
        if 'x' not in data or 'y' not in data or 'z' not in data:
            return False, 0, 0, "Missing x, y, or z coordinates"
        
        if 'semantic_seg' not in data:
            return False, 0, 0, "Missing semantic_seg field"
        
        if 'treeID' not in data:
            return False, 0, 0, "Missing treeID field"
        
        x = data['x'].astype(np.float64)
        y = data['y'].astype(np.float64)
        z = data['z'].astype(np.float64)
        semantic_seg = data['semantic_seg'].astype(np.int64)
        treeID = data['treeID'].astype(np.int64)
        
        # Step 3: Fix semantic label mapping
        # Model expects: 0=ground, 1=wood, 2=leaf
        # Check current values
        unique_sem = np.unique(semantic_seg)
        print(f"  Current semantic_seg values: {unique_sem}")
        
        # Fix mapping: if we have 2,3, remap to 1,2
        # If we have 1,2, keep them (but they should be wood,leaf, not ground,wood)
        needs_fix = False
        if np.any(semantic_seg == 2) or np.any(semantic_seg == 3):
            # Old mapping: 2=wood, 3=leaf
            # New mapping: 1=wood, 2=leaf
            wood_mask = (semantic_seg == 2)
            leaf_mask = (semantic_seg == 3)
            semantic_seg[wood_mask] = 1  # wood
            semantic_seg[leaf_mask] = 2  # leaf
            needs_fix = True
            print(f"  Fixed mapping: 2→1 (wood), 3→2 (leaf)")
        elif np.any(semantic_seg == 1) or np.any(semantic_seg == 2):
            # Check if 1 is actually wood (not ground)
            # If we have 1,2 but no 0, assume 1=wood, 2=leaf (already correct)
            if 0 not in unique_sem:
                print(f"  Labels already correct (1=wood, 2=leaf), but no ground points")
            else:
                print(f"  Labels appear correct (0=ground, 1=wood, 2=leaf)")
        
        # Step 4: Add dummy ground points at the bottom
        # Calculate bounding box
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        
        # Create ground points in a grid pattern at the bottom
        # Use a slightly larger area than the point cloud
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Create grid of ground points
        grid_size = int(np.sqrt(num_ground_points))
        if grid_size < 1:
            grid_size = 1
        
        # Generate grid coordinates
        x_ground = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, grid_size)
        y_ground = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, grid_size)
        xx, yy = np.meshgrid(x_ground, y_ground)
        
        # Flatten and take only the requested number
        x_ground_flat = xx.flatten()[:num_ground_points]
        y_ground_flat = yy.flatten()[:num_ground_points]
        z_ground = np.full(len(x_ground_flat), z_min + ground_height_offset, dtype=np.float64)
        
        # Ground points have semantic_seg = 0 and treeID = 0 (background)
        semantic_seg_ground = np.zeros(len(x_ground_flat), dtype=np.int64)
        treeID_ground = np.zeros(len(x_ground_flat), dtype=np.int64)
        
        # Step 5: Concatenate original points with ground points
        x_all = np.concatenate([x, x_ground_flat])
        y_all = np.concatenate([y, y_ground_flat])
        z_all = np.concatenate([z, z_ground])
        semantic_seg_all = np.concatenate([semantic_seg, semantic_seg_ground])
        treeID_all = np.concatenate([treeID, treeID_ground])
        
        num_total = len(x_all)
        num_ground_added = len(x_ground_flat)
        
        # Step 6: Create new PLY with fixed data
        dtype_list = [
            ('x', 'f8'),      # float64
            ('y', 'f8'),
            ('z', 'f8'),
            ('semantic_seg', 'i4'),  # int32
            ('treeID', 'i4')
        ]
        
        vertex_data = np.empty(num_total, dtype=dtype_list)
        vertex_data['x'] = x_all
        vertex_data['y'] = y_all
        vertex_data['z'] = z_all
        vertex_data['semantic_seg'] = semantic_seg_all
        vertex_data['treeID'] = treeID_all
        
        # Step 7: Write binary PLY file
        el = PlyElement.describe(vertex_data, 'vertex')
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        PlyData([el], text=False).write(output_path)
        
        # Count points by class
        unique_sem_final = np.unique(semantic_seg_all)
        counts = {int(sem): int(np.sum(semantic_seg_all == sem)) for sem in unique_sem_final}
        
        return True, num_total, num_ground_added, f"Fixed: {num_points} original + {num_ground_added} ground = {num_total} total. Classes: {counts}"
        
    except Exception as e:
        return False, 0, 0, f"Error: {str(e)}"


def process_directory(input_dir, output_dir=None, num_ground_points=1000, ground_height_offset=-0.5):
    """
    Process all PLY files in input directory.
    
    Args:
        input_dir: Input directory containing PLY files
        output_dir: Output directory (if None, overwrites input files)
        num_ground_points: Number of dummy ground points to add per file
        ground_height_offset: Z offset for ground points
    """
    input_dir = os.path.abspath(input_dir)
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all PLY files
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    
    if not ply_files:
        print(f"⚠️  No PLY files found in {input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY file(s) to process")
    print(f"Input directory: {input_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print(f"⚠️  Will overwrite input files!")
    print(f"Ground points per file: {num_ground_points}")
    print("-" * 80)
    
    successful = []
    failed = []
    
    for i, input_file in enumerate(ply_files, 1):
        base_name = os.path.basename(input_file)
        if output_dir:
            output_file = os.path.join(output_dir, base_name)
        else:
            output_file = input_file
        
        print(f"\n[{i}/{len(ply_files)}] Processing: {base_name}")
        
        success, num_points, num_ground, message = fix_ply_file(
            input_file, 
            output_file,
            num_ground_points=num_ground_points,
            ground_height_offset=ground_height_offset
        )
        
        if success:
            print(f"  ✅ {message}")
            if output_dir:
                print(f"  Saved: {os.path.relpath(output_file, output_dir)}")
            successful.append((base_name, num_points, num_ground))
        else:
            print(f"  ❌ Failed: {message}")
            failed.append((base_name, message))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(successful)}/{len(ply_files)}")
    if successful:
        total_points = sum(points for _, points, _ in successful)
        total_ground = sum(ground for _, _, ground in successful)
        print(f"   Total points: {total_points:,} (added {total_ground:,} ground points)")
        for name, points, ground in successful:
            print(f"   - {name}: {points:,} points ({ground:,} ground)")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(ply_files)}")
        for name, reason in failed:
            print(f"   - {name}: {reason}")
    
    if output_dir:
        print(f"\n✅ Fixed files saved to: {output_dir}")
    else:
        print(f"\n✅ Files updated in place: {input_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fix PLY files by correcting semantic labels and adding ground points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix files in place (overwrites original files)
  python tools/fix_ply_labels_and_add_ground.py --input-dir data/ForAINetV2/train_val_data
  
  # Fix files and save to new directory
  python tools/fix_ply_labels_and_add_ground.py --input-dir data/ForAINetV2/train_val_data --output-dir data/ForAINetV2/train_val_data_fixed
  
  # Add more ground points
  python tools/fix_ply_labels_and_add_ground.py --input-dir data/ForAINetV2/train_val_data --num-ground-points 5000
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing PLY files to fix'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (if not specified, overwrites input files)'
    )
    
    parser.add_argument(
        '--num-ground-points',
        type=int,
        default=1000,
        help='Number of dummy ground points to add per file (default: 1000)'
    )
    
    parser.add_argument(
        '--ground-height-offset',
        type=float,
        default=-0.5,
        help='Z offset for ground points relative to min Z (default: -0.5)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    process_directory(
        args.input_dir,
        args.output_dir,
        num_ground_points=args.num_ground_points,
        ground_height_offset=args.ground_height_offset
    )


