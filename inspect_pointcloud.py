#!/usr/bin/env python3
"""
Simple script to inspect PLY and LAS point cloud files.
Displays fields and basic statistics for each feature.

Usage:
    python inspect_pointcloud.py <file.ply>
    python inspect_pointcloud.py <file.las>
    python inspect_pointcloud.py test_files/130_labeled.ply
"""

import sys
import os
import numpy as np

# Conditional imports
try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    print("Warning: plyfile not available. PLY file inspection will not work.")

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    print("Warning: laspy not available. LAS file inspection will not work.")


def format_number(value, dtype):
    """Format number based on its dtype for better readability."""
    if np.issubdtype(dtype, np.integer):
        return f"{int(value):,}"
    elif np.issubdtype(dtype, np.floating):
        return f"{value:.6f}"
    else:
        return str(value)


def get_statistics(data, field_name, dtype):
    """Calculate and return statistics for a field."""
    stats = {}
    
    # Basic info - store both string representation and actual dtype
    stats['dtype'] = dtype  # Store actual dtype for format_number
    # Clean dtype string representation
    if hasattr(dtype, 'name'):
        stats['dtype_str'] = dtype.name
    elif isinstance(dtype, type):
        stats['dtype_str'] = dtype.__name__
    else:
        stats['dtype_str'] = str(dtype)
    stats['shape'] = data.shape
    stats['count'] = len(data)
    
    # Check if numeric
    if np.issubdtype(dtype, np.number):
        stats['min'] = np.min(data)
        stats['max'] = np.max(data)
        stats['mean'] = np.mean(data)
        stats['std'] = np.std(data)
        stats['median'] = np.median(data)
        
        # Count unique values (useful for categorical data)
        unique_vals = np.unique(data)
        stats['unique_count'] = len(unique_vals)
        
        # If few unique values, show them
        if len(unique_vals) <= 20:
            stats['unique_values'] = unique_vals.tolist()
        else:
            stats['unique_values'] = f"{len(unique_vals)} unique values"
        
        # Check for NaN/Inf
        if np.issubdtype(dtype, np.floating):
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            if nan_count > 0:
                stats['nan_count'] = nan_count
            if inf_count > 0:
                stats['inf_count'] = inf_count
    else:
        # Non-numeric data
        unique_vals = np.unique(data)
        stats['unique_count'] = len(unique_vals)
        if len(unique_vals) <= 20:
            stats['unique_values'] = unique_vals.tolist()
        else:
            stats['unique_values'] = f"{len(unique_vals)} unique values"
    
    return stats


def inspect_ply(file_path):
    """Inspect a PLY file."""
    if not PLYFILE_AVAILABLE:
        print("Error: plyfile library is not available. Please install it: pip install plyfile")
        return
    
    print(f"\n{'='*80}")
    print(f"Inspecting PLY file: {file_path}")
    print(f"{'='*80}\n")
    
    try:
        plydata = PlyData.read(file_path)
        
        # Get vertex element (point cloud data)
        # plydata.elements is a list of PlyElement objects
        vertex = None
        for elem in plydata.elements:
            if elem.name == 'vertex':
                vertex = elem
                break
        
        if vertex is None:
            print("Warning: No 'vertex' element found in PLY file.")
            print(f"Available elements: {[elem.name for elem in plydata.elements]}")
            return
        
        # Access data directly from the element
        vertex_data = vertex.data
        num_points = len(vertex_data)
        
        print(f"Number of points: {num_points:,}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"\nFields ({len(vertex_data.dtype.names)}):")
        print("-" * 80)
        
        # Get all field names and their data
        field_stats = {}
        for field_name in vertex_data.dtype.names:
            field_data = vertex_data[field_name]
            dtype = field_data.dtype
            stats = get_statistics(field_data, field_name, dtype)
            field_stats[field_name] = stats
        
        # Display statistics for each field
        for field_name, stats in field_stats.items():
            print(f"\n  Field: {field_name}")
            print(f"    Type: {stats['dtype_str']}")
            print(f"    Count: {stats['count']:,}")
            
            if 'min' in stats:
                print(f"    Min: {format_number(stats['min'], stats['dtype'])}")
                print(f"    Max: {format_number(stats['max'], stats['dtype'])}")
                print(f"    Mean: {format_number(stats['mean'], stats['dtype'])}")
                print(f"    Std: {format_number(stats['std'], stats['dtype'])}")
                print(f"    Median: {format_number(stats['median'], stats['dtype'])}")
            
            print(f"    Unique values: {stats['unique_count']}")
            if isinstance(stats['unique_values'], list):
                print(f"      Values: {stats['unique_values']}")
            
            if 'nan_count' in stats:
                print(f"    NaN count: {stats['nan_count']:,}")
            if 'inf_count' in stats:
                print(f"    Inf count: {stats['inf_count']:,}")
        
        # Check for other elements (like faces)
        other_elements = [elem for elem in plydata.elements if elem.name != 'vertex']
        if other_elements:
            print(f"\nOther elements found: {[elem.name for elem in other_elements]}")
            for elem in other_elements:
                print(f"  {elem.name}: {len(elem.data)} items")
        
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        import traceback
        traceback.print_exc()


def inspect_las(file_path):
    """Inspect a LAS file."""
    if not LASPY_AVAILABLE:
        print("Error: laspy library is not available. Please install it: pip install laspy")
        return
    
    print(f"\n{'='*80}")
    print(f"Inspecting LAS file: {file_path}")
    print(f"{'='*80}\n")
    
    try:
        las = laspy.read(file_path)
        num_points = len(las.points)
        
        print(f"Number of points: {num_points:,}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"LAS Version: {las.header.version}")
        print(f"Point Format: {las.header.point_format.id}")
        print(f"\nFields:")
        print("-" * 80)
        
        # Get all available fields - check common LAS attributes
        all_fields = [
            ('x', las.x, np.float64),
            ('y', las.y, np.float64),
            ('z', las.z, np.float64),
        ]
        
        # Common LAS field names to check
        common_fields = ['intensity', 'return_number', 'number_of_returns', 'classification',
                        'scan_angle_rank', 'scan_angle', 'user_data', 'point_source_id',
                        'gps_time', 'red', 'green', 'blue', 'edge_of_flight_line',
                        'scan_direction_flag', 'synthetic', 'key_point', 'withheld', 'overlap']
        
        for field_name in common_fields:
            if hasattr(las, field_name):
                field_data = getattr(las, field_name)
                all_fields.append((field_name, field_data, field_data.dtype))
        
        # Extra dimensions (LAS 1.4+)
        if hasattr(las, 'extra_dims') and las.extra_dims:
            for dim_name in las.extra_dims:
                all_fields.append((dim_name, getattr(las, dim_name), getattr(las, dim_name).dtype))
        
        # Display statistics for each field
        for field_name, field_data, dtype in all_fields:
            # Convert to numpy array
            data = np.array(field_data)
            stats = get_statistics(data, field_name, dtype)
            
            print(f"\n  Field: {field_name}")
            print(f"    Type: {stats['dtype_str']}")
            print(f"    Count: {stats['count']:,}")
            
            if 'min' in stats:
                print(f"    Min: {format_number(stats['min'], stats['dtype'])}")
                print(f"    Max: {format_number(stats['max'], stats['dtype'])}")
                print(f"    Mean: {format_number(stats['mean'], stats['dtype'])}")
                print(f"    Std: {format_number(stats['std'], stats['dtype'])}")
                print(f"    Median: {format_number(stats['median'], stats['dtype'])}")
            
            print(f"    Unique values: {stats['unique_count']}")
            if isinstance(stats['unique_values'], list):
                print(f"      Values: {stats['unique_values']}")
            
            if 'nan_count' in stats:
                print(f"    NaN count: {stats['nan_count']:,}")
            if 'inf_count' in stats:
                print(f"    Inf count: {stats['inf_count']:,}")
        
        # Header information
        print(f"\nHeader Information:")
        print(f"  Scale: X={las.header.x_scale}, Y={las.header.y_scale}, Z={las.header.z_scale}")
        print(f"  Offset: X={las.header.x_offset}, Y={las.header.y_offset}, Z={las.header.z_offset}")
        print(f"  Bounds: X=[{las.header.x_min:.3f}, {las.header.x_max:.3f}], "
              f"Y=[{las.header.y_min:.3f}, {las.header.y_max:.3f}], "
              f"Z=[{las.header.z_min:.3f}, {las.header.z_max:.3f}]")
        
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pointcloud.py <file.ply|file.las>")
        print("\nExample:")
        print("  python inspect_pointcloud.py test_files/130_labeled.ply")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.ply':
        inspect_ply(file_path)
    elif file_ext in ['.las', '.laz']:
        inspect_las(file_path)
    else:
        print(f"Error: Unsupported file type: {file_ext}")
        print("Supported formats: .ply, .las, .laz")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

