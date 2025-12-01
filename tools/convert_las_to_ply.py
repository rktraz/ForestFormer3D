#!/usr/bin/env python3
"""
Convert LAS/LAZ files to PLY format preserving all original fields.

Usage:
    python tools/convert_las_to_ply.py input.las output.ply
    python tools/convert_las_to_ply.py input.laz output.ply
"""

import sys
import os
import numpy as np
from plyfile import PlyData, PlyElement
import laspy


def convert_las_to_ply_full(input_path, output_path):
    """
    Convert LAS/LAZ file to PLY format preserving all original fields.
    
    Args:
        input_path: Path to input LAS/LAZ file
        output_path: Path to output PLY file
    """
    print(f"Reading LAS file: {input_path}")
    
    # Read LAS file
    las = laspy.read(input_path)
    
    # Extract coordinates (laspy automatically applies scaling and offsets)
    # Convert to numpy arrays explicitly with float64 for precision
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    
    num_points = len(x)
    print(f"Found {num_points:,} points")
    print(f"Coordinate ranges: X=[{x.min():.3f}, {x.max():.3f}], Y=[{y.min():.3f}, {y.max():.3f}], Z=[{z.min():.3f}, {z.max():.3f}]")
    
    # Build dtype list for PLY with all available fields
    dtype_list = []
    field_data = {}
    
    # Standard fields - always present (use float64 for better precision)
    dtype_list.append(('x', 'f8'))
    dtype_list.append(('y', 'f8'))
    dtype_list.append(('z', 'f8'))
    field_data['x'] = x
    field_data['y'] = y
    field_data['z'] = z
    
    # Intensity
    if hasattr(las, 'intensity'):
        dtype_list.append(('intensity', 'u2'))
        field_data['intensity'] = np.array(las.intensity, dtype=np.uint16)
    
    # Return number and number of returns
    if hasattr(las, 'return_number'):
        dtype_list.append(('return_number', 'u1'))
        field_data['return_number'] = np.array(las.return_number, dtype=np.uint8)
    
    if hasattr(las, 'number_of_returns'):
        dtype_list.append(('number_of_returns', 'u1'))
        field_data['number_of_returns'] = np.array(las.number_of_returns, dtype=np.uint8)
    
    # Classification
    if hasattr(las, 'classification'):
        dtype_list.append(('classification', 'u1'))
        field_data['classification'] = np.array(las.classification, dtype=np.uint8)
    
    # Scan angle
    if hasattr(las, 'scan_angle_rank'):
        dtype_list.append(('scan_angle_rank', 'i1'))
        field_data['scan_angle_rank'] = np.array(las.scan_angle_rank, dtype=np.int8)
    elif hasattr(las, 'scan_angle'):
        dtype_list.append(('scan_angle', 'i2'))
        field_data['scan_angle'] = np.array(las.scan_angle, dtype=np.int16)
    
    # User data
    if hasattr(las, 'user_data'):
        dtype_list.append(('user_data', 'u1'))
        field_data['user_data'] = np.array(las.user_data, dtype=np.uint8)
    
    # Point source ID
    if hasattr(las, 'point_source_id'):
        dtype_list.append(('point_source_id', 'u2'))
        field_data['point_source_id'] = np.array(las.point_source_id, dtype=np.uint16)
    
    # GPS time
    if hasattr(las, 'gps_time'):
        dtype_list.append(('gps_time', 'f8'))
        field_data['gps_time'] = np.array(las.gps_time, dtype=np.float64)
    
    # RGB colors
    if hasattr(las, 'red'):
        dtype_list.append(('red', 'u2'))
        field_data['red'] = np.array(las.red, dtype=np.uint16)
    
    if hasattr(las, 'green'):
        dtype_list.append(('green', 'u2'))
        field_data['green'] = np.array(las.green, dtype=np.uint16)
    
    if hasattr(las, 'blue'):
        dtype_list.append(('blue', 'u2'))
        field_data['blue'] = np.array(las.blue, dtype=np.uint16)
    
    # Edge of flight line
    if hasattr(las, 'edge_of_flight_line'):
        dtype_list.append(('edge_of_flight_line', 'u1'))
        field_data['edge_of_flight_line'] = np.array(las.edge_of_flight_line, dtype=np.uint8)
    
    # Scan direction flag
    if hasattr(las, 'scan_direction_flag'):
        dtype_list.append(('scan_direction_flag', 'u1'))
        field_data['scan_direction_flag'] = np.array(las.scan_direction_flag, dtype=np.uint8)
    
    # Synthetic flag
    if hasattr(las, 'synthetic'):
        dtype_list.append(('synthetic', 'u1'))
        field_data['synthetic'] = np.array(las.synthetic, dtype=np.uint8)
    
    # Key point flag
    if hasattr(las, 'key_point'):
        dtype_list.append(('key_point', 'u1'))
        field_data['key_point'] = np.array(las.key_point, dtype=np.uint8)
    
    # Withheld flag
    if hasattr(las, 'withheld'):
        dtype_list.append(('withheld', 'u1'))
        field_data['withheld'] = np.array(las.withheld, dtype=np.uint8)
    
    # Overlap flag
    if hasattr(las, 'overlap'):
        dtype_list.append(('overlap', 'u1'))
        field_data['overlap'] = np.array(las.overlap, dtype=np.uint8)
    
    # Extra dimensions (LAS 1.4+)
    if hasattr(las, 'extra_dims'):
        for dim_name, dim_info in las.extra_dims.items():
            # Map LAS data types to numpy/PLY compatible types
            las_type = dim_info.type
            if las_type == 'uint8':
                ply_type = 'u1'
            elif las_type == 'uint16':
                ply_type = 'u2'
            elif las_type == 'uint32':
                ply_type = 'u4'
            elif las_type == 'uint64':
                ply_type = 'u8'
            elif las_type == 'int8':
                ply_type = 'i1'
            elif las_type == 'int16':
                ply_type = 'i2'
            elif las_type == 'int32':
                ply_type = 'i4'
            elif las_type == 'int64':
                ply_type = 'i8'
            elif las_type == 'float32':
                ply_type = 'f4'
            elif las_type == 'float64':
                ply_type = 'f8'
            else:
                # Default to float32 for unknown types
                ply_type = 'f4'
                print(f"Warning: Unknown type {las_type} for dimension {dim_name}, using f4")
            
            dtype_list.append((dim_name, ply_type))
            # Convert to numpy array with appropriate dtype
            dim_data = getattr(las, dim_name)
            if ply_type == 'u1':
                field_data[dim_name] = np.array(dim_data, dtype=np.uint8)
            elif ply_type == 'u2':
                field_data[dim_name] = np.array(dim_data, dtype=np.uint16)
            elif ply_type == 'u4':
                field_data[dim_name] = np.array(dim_data, dtype=np.uint32)
            elif ply_type == 'u8':
                field_data[dim_name] = np.array(dim_data, dtype=np.uint64)
            elif ply_type == 'i1':
                field_data[dim_name] = np.array(dim_data, dtype=np.int8)
            elif ply_type == 'i2':
                field_data[dim_name] = np.array(dim_data, dtype=np.int16)
            elif ply_type == 'i4':
                field_data[dim_name] = np.array(dim_data, dtype=np.int32)
            elif ply_type == 'i8':
                field_data[dim_name] = np.array(dim_data, dtype=np.int64)
            elif ply_type == 'f4':
                field_data[dim_name] = np.array(dim_data, dtype=np.float32)
            elif ply_type == 'f8':
                field_data[dim_name] = np.array(dim_data, dtype=np.float64)
            else:
                field_data[dim_name] = np.array(dim_data)
    
    # Create structured array for PLY
    print(f"Preserving {len(dtype_list)} fields: {[d[0] for d in dtype_list]}")
    vertex_data = np.empty(num_points, dtype=dtype_list)
    
    # Fill in all the fields
    for field_name, field_values in field_data.items():
        vertex_data[field_name] = field_values
    
    # Create PLY element
    el = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    print(f"Writing PLY file: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    PlyData([el], text=False).write(output_path)
    
    print(f"✅ Successfully converted {num_points:,} points to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"   Preserved {len(dtype_list)} fields")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_las_to_ply.py <input.las> <output.ply>")
        print("\nThis script converts LAS/LAZ files to PLY format while preserving")
        print("all original fields including coordinates, intensity, classification,")
        print("RGB colors, GPS time, and any extra dimensions.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.lower().endswith(('.las', '.laz')):
        print(f"Warning: Input file doesn't have .las or .laz extension: {input_path}")
    
    try:
        convert_las_to_ply_full(input_path, output_path)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

