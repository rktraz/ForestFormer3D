#!/usr/bin/env python3
"""
Convert LAS/LAZ files to PLY format for ForestFormer3D inference.

Usage:
    python tools/convert_las_to_ply.py input.las output.ply
    python tools/convert_las_to_ply.py input.laz output.ply
"""

import sys
import os
import numpy as np
from plyfile import PlyData, PlyElement
import laspy

def convert_las_to_ply(input_path, output_path):
    """
    Convert LAS/LAZ file to PLY format.
    
    Args:
        input_path: Path to input LAS/LAZ file
        output_path: Path to output PLY file
    """
    print(f"Reading LAS file: {input_path}")
    
    # Read LAS file
    las = laspy.read(input_path)
    
    # Extract coordinates
    x = las.x
    y = las.y
    z = las.z
    
    num_points = len(x)
    print(f"Found {num_points:,} points")
    
    # Create structured array for PLY
    vertex_data = np.empty(num_points, dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4')
    ])
    
    vertex_data['x'] = x
    vertex_data['y'] = y
    vertex_data['z'] = z
    
    # Create PLY element
    el = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    print(f"Writing PLY file: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    PlyData([el], text=False).write(output_path)
    
    print(f"✅ Successfully converted {num_points:,} points to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_las_to_ply.py <input.las> <output.ply>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.lower().endswith(('.las', '.laz')):
        print(f"Warning: Input file doesn't have .las or .laz extension: {input_path}")
    
    convert_las_to_ply(input_path, output_path)

