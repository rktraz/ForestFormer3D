#!/usr/bin/env python3
"""
Standardize point cloud files (LAS/PLY) to have consistent field names and data types.

This script:
1. Reads LAS or PLY files
2. Standardizes field names (removes scalar_ prefix, uses lowercase)
3. Normalizes data types (uint16/uint8 for integers, float32 for floats)
4. Preserves all useful scanner features
5. Outputs consistent PLY files

Usage:
    python tools/standardize_pointcloud_data.py --input-dir /path/to/input --output-dir /path/to/output
    python tools/standardize_pointcloud_data.py --input-dir /path/to/input --output-dir /path/to/output --preserve-labels
"""

import os
import sys
import glob
import argparse
import subprocess
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

# Conditional imports
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    print("Warning: laspy not available. LAS file conversion will not work.")


# Field name mapping: from various formats to standardized names
FIELD_NAME_MAPPING = {
    # Coordinates (always present)
    'x': 'x',
    'y': 'y',
    'z': 'z',
    
    # Scanner features - standardize to lowercase, remove scalar_ prefix
    'intensity': 'intensity',
    'scalar_intensity': 'intensity',
    'scalar_Intensity': 'intensity',
    
    'return_number': 'return_number',
    'scalar_return_number': 'return_number',
    'scalar_Return_Number': 'return_number',
    
    'number_of_returns': 'number_of_returns',
    'scalar_number_of_returns': 'number_of_returns',
    'scalar_Number_Of_Returns': 'number_of_returns',
    
    'classification': 'classification',
    'scalar_classification': 'classification',
    'scalar_Classification': 'classification',
    
    'scan_angle': 'scan_angle',
    'scan_angle_rank': 'scan_angle_rank',
    'scalar_scan_angle': 'scan_angle',
    'scalar_Scan_Angle': 'scan_angle',
    
    'user_data': 'user_data',
    'scalar_user_data': 'user_data',
    'scalar_User_Data': 'user_data',
    
    'point_source_id': 'point_source_id',
    'scalar_point_source_id': 'point_source_id',
    'scalar_Point_Source_ID': 'point_source_id',
    
    'gps_time': 'gps_time',
    'scalar_gps_time': 'gps_time',
    'scalar_Gps_Time': 'gps_time',
    
    # RGB colors
    'red': 'red',
    'green': 'green',
    'blue': 'blue',
    
    # Flags (usually constant, but preserve if present)
    'edge_of_flight_line': 'edge_of_flight_line',
    'scan_direction_flag': 'scan_direction_flag',
    'synthetic': 'synthetic',
    'key_point': 'key_point',
    'withheld': 'withheld',
    'overlap': 'overlap',
    'scanner_channel': 'scanner_channel',
    
    # Labels (if present)
    'semantic_seg': 'semantic_seg',
    'treeID': 'treeID',
    'tree_id': 'treeID',
}

# Standardized data types for each field
STANDARD_DTYPES = {
    'x': 'f8',  # float64 for coordinates
    'y': 'f8',
    'z': 'f8',
    'intensity': 'u2',  # uint16
    'return_number': 'u1',  # uint8
    'number_of_returns': 'u1',  # uint8
    'classification': 'u1',  # uint8
    'scan_angle': 'i2',  # int16
    'scan_angle_rank': 'i1',  # int8
    'user_data': 'u1',  # uint8
    'point_source_id': 'u2',  # uint16
    'gps_time': 'f8',  # float64
    'red': 'u2',  # uint16 (normalize from uint8 if needed)
    'green': 'u2',
    'blue': 'u2',
    'edge_of_flight_line': 'u1',
    'scan_direction_flag': 'u1',
    'synthetic': 'u1',
    'key_point': 'u1',
    'withheld': 'u1',
    'overlap': 'u1',
    'scanner_channel': 'u1',
    'semantic_seg': 'i4',  # int32
    'treeID': 'i4',  # int32
}


def normalize_field_name(field_name):
    """Normalize field name to standard format."""
    field_lower = field_name.lower()
    return FIELD_NAME_MAPPING.get(field_name, FIELD_NAME_MAPPING.get(field_lower, field_name.lower()))


def read_las_file(las_path):
    """Read LAS file and return standardized data dictionary."""
    if not LASPY_AVAILABLE:
        raise RuntimeError("laspy is required to read LAS files. Install with: pip install laspy")
    
    las = laspy.read(las_path)
    num_points = len(las.points)
    
    data = {}
    
    # Coordinates
    data['x'] = np.array(las.x, dtype=np.float64)
    data['y'] = np.array(las.y, dtype=np.float64)
    data['z'] = np.array(las.z, dtype=np.float64)
    
    # Scanner features
    if hasattr(las, 'intensity'):
        data['intensity'] = np.array(las.intensity, dtype=np.uint16)
    
    if hasattr(las, 'return_number'):
        data['return_number'] = np.array(las.return_number, dtype=np.uint8)
    
    if hasattr(las, 'number_of_returns'):
        data['number_of_returns'] = np.array(las.number_of_returns, dtype=np.uint8)
    
    if hasattr(las, 'classification'):
        data['classification'] = np.array(las.classification, dtype=np.uint8)
    
    # Scan angle (can be scan_angle_rank or scan_angle)
    if hasattr(las, 'scan_angle'):
        data['scan_angle'] = np.array(las.scan_angle, dtype=np.int16)
    elif hasattr(las, 'scan_angle_rank'):
        data['scan_angle_rank'] = np.array(las.scan_angle_rank, dtype=np.int8)
    
    if hasattr(las, 'user_data'):
        data['user_data'] = np.array(las.user_data, dtype=np.uint8)
    
    if hasattr(las, 'point_source_id'):
        data['point_source_id'] = np.array(las.point_source_id, dtype=np.uint16)
    
    if hasattr(las, 'gps_time'):
        gps_time = np.array(las.gps_time, dtype=np.float64)
        # Filter out invalid values (inf, -inf, very large numbers)
        valid_mask = np.isfinite(gps_time) & (np.abs(gps_time) < 1e10)
        if np.any(valid_mask):
            data['gps_time'] = gps_time
        else:
            print(f"  Warning: All GPS time values are invalid, skipping")
    
    # RGB colors
    if hasattr(las, 'red'):
        # LAS RGB is typically uint16, scale to 0-65535 range
        data['red'] = np.array(las.red, dtype=np.uint16)
        data['green'] = np.array(las.green, dtype=np.uint16)
        data['blue'] = np.array(las.blue, dtype=np.uint16)
    
    # Flags (usually constant, but preserve)
    if hasattr(las, 'edge_of_flight_line'):
        data['edge_of_flight_line'] = np.array(las.edge_of_flight_line, dtype=np.uint8)
    
    if hasattr(las, 'scan_direction_flag'):
        data['scan_direction_flag'] = np.array(las.scan_direction_flag, dtype=np.uint8)
    
    if hasattr(las, 'synthetic'):
        data['synthetic'] = np.array(las.synthetic, dtype=np.uint8)
    
    if hasattr(las, 'key_point'):
        data['key_point'] = np.array(las.key_point, dtype=np.uint8)
    
    if hasattr(las, 'withheld'):
        data['withheld'] = np.array(las.withheld, dtype=np.uint8)
    
    if hasattr(las, 'overlap'):
        data['overlap'] = np.array(las.overlap, dtype=np.uint8)
    
    # Extra dimensions (LAS 1.4+)
    if hasattr(las, 'extra_dims'):
        for dim_name, dim_info in las.extra_dims.items():
            dim_data = getattr(las, dim_name)
            # Map to appropriate numpy type
            if dim_info.type == 'uint8':
                data[dim_name] = np.array(dim_data, dtype=np.uint8)
            elif dim_info.type == 'uint16':
                data[dim_name] = np.array(dim_data, dtype=np.uint16)
            elif dim_info.type == 'float32':
                data[dim_name] = np.array(dim_data, dtype=np.float32)
            elif dim_info.type == 'float64':
                data[dim_name] = np.array(dim_data, dtype=np.float64)
            else:
                data[dim_name] = np.array(dim_data)
    
    return data, num_points


def read_ply_file(ply_path):
    """Read PLY file and return standardized data dictionary."""
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data
    
    data = {}
    for field_name in vertex_data.dtype.names:
        field_array = np.asarray(vertex_data[field_name])
        
        # Skip fields with all NaN values
        if np.issubdtype(field_array.dtype, np.floating):
            if np.all(np.isnan(field_array)):
                continue
        
        # Normalize field name
        std_name = normalize_field_name(field_name)
        
        # Convert data type to standard
        if std_name in STANDARD_DTYPES:
            target_dtype = STANDARD_DTYPES[std_name]
            
            # Handle type conversions
            if target_dtype == 'u2' and field_array.dtype == np.uint8:
                # RGB: uint8 -> uint16 (scale by 256)
                if std_name in ['red', 'green', 'blue']:
                    data[std_name] = (field_array.astype(np.uint16) * 256).clip(0, 65535)
                else:
                    data[std_name] = field_array.astype(np.uint16)
            elif target_dtype == 'f8' and np.issubdtype(field_array.dtype, np.floating):
                # Float conversion
                data[std_name] = field_array.astype(np.float64)
            elif target_dtype == 'u1' and np.issubdtype(field_array.dtype, np.integer):
                # Integer to uint8
                data[std_name] = field_array.astype(np.uint8).clip(0, 255)
            elif target_dtype == 'u2' and np.issubdtype(field_array.dtype, np.integer):
                # Integer to uint16
                data[std_name] = field_array.astype(np.uint16).clip(0, 65535)
            elif target_dtype == 'i2' and np.issubdtype(field_array.dtype, np.integer):
                # Integer to int16
                data[std_name] = field_array.astype(np.int16).clip(-32768, 32767)
            elif target_dtype == 'i4' and np.issubdtype(field_array.dtype, np.integer):
                # Integer to int32
                data[std_name] = field_array.astype(np.int32)
            else:
                # Try direct conversion
                try:
                    data[std_name] = field_array.astype(np.dtype(target_dtype))
                except:
                    print(f"  Warning: Could not convert {field_name} to {target_dtype}, keeping original type")
                    data[std_name] = field_array
        else:
            # Unknown field, keep as is but normalize name
            data[std_name] = field_array
    
    num_points = len(vertex_data)
    return data, num_points


def standardize_file(input_path, output_path, preserve_labels=False):
    """
    Standardize a single file (LAS or PLY) to consistent format.
    
    Args:
        input_path: Path to input file (LAS/LAZ/PLY)
        output_path: Path to output PLY file
        preserve_labels: If True, preserve semantic_seg and treeID if present
        
    Returns:
        tuple: (success: bool, num_points: int, message: str)
    """
    input_ext = os.path.splitext(input_path)[1].lower()
    temp_ply = None
    
    try:
        # Step 1: Read file
        if input_ext in ['.las', '.laz']:
            if not LASPY_AVAILABLE:
                return False, 0, "laspy not available. Install with: pip install laspy"
            data, num_points = read_las_file(input_path)
        elif input_ext == '.ply':
            data, num_points = read_ply_file(input_path)
        else:
            return False, 0, f"Unsupported file format: {input_ext}"
        
        if num_points == 0:
            return False, 0, "File contains no points"
        
        # Step 2: Ensure coordinates are present
        if 'x' not in data or 'y' not in data or 'z' not in data:
            return False, 0, "Missing x, y, or z coordinates"
        
        # Step 3: Handle labels if preserve_labels is True
        if preserve_labels:
            # Check if labels exist, if not create default
            if 'semantic_seg' not in data:
                # Try to infer from classification
                if 'classification' in data:
                    # Map: 0=wood→1, 1=leaf→2, add ground later
                    semantic_seg = np.zeros(num_points, dtype=np.int32)
                    semantic_seg[data['classification'] == 0] = 1  # wood
                    semantic_seg[data['classification'] == 1] = 2  # leaf
                    data['semantic_seg'] = semantic_seg
                else:
                    # Default: all wood
                    data['semantic_seg'] = np.ones(num_points, dtype=np.int32)
            
            if 'treeID' not in data:
                # Default: all points belong to tree 1
                data['treeID'] = np.ones(num_points, dtype=np.int32)
        
        # Step 4: Build dtype list for output PLY
        # Always include coordinates first
        dtype_list = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        field_order = ['x', 'y', 'z']
        
        # Add scanner features in consistent order
        scanner_features = [
            'intensity', 'return_number', 'number_of_returns', 'classification',
            'scan_angle', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time'
        ]
        
        for feature in scanner_features:
            if feature in data:
                std_dtype = STANDARD_DTYPES.get(feature, 'f4')
                dtype_list.append((feature, std_dtype))
                field_order.append(feature)
        
        # Add RGB colors
        if 'red' in data and 'green' in data and 'blue' in data:
            dtype_list.append(('red', 'u2'))
            dtype_list.append(('green', 'u2'))
            dtype_list.append(('blue', 'u2'))
            field_order.extend(['red', 'green', 'blue'])
        
        # Add flags (if not all constant)
        flags = ['edge_of_flight_line', 'scan_direction_flag', 'synthetic', 
                 'key_point', 'withheld', 'overlap', 'scanner_channel']
        for flag in flags:
            if flag in data:
                unique_vals = np.unique(data[flag])
                if len(unique_vals) > 1:  # Only include if not constant
                    dtype_list.append((flag, 'u1'))
                    field_order.append(flag)
        
        # Add labels if preserve_labels
        if preserve_labels:
            if 'semantic_seg' in data:
                dtype_list.append(('semantic_seg', 'i4'))
                field_order.append('semantic_seg')
            if 'treeID' in data:
                dtype_list.append(('treeID', 'i4'))
                field_order.append('treeID')
        
        # Step 5: Create structured array
        vertex_data = np.empty(num_points, dtype=dtype_list)
        
        for field_name in field_order:
            if field_name in data:
                vertex_data[field_name] = data[field_name]
            else:
                # Fill with zeros/defaults if missing
                if field_name in ['red', 'green', 'blue']:
                    vertex_data[field_name] = 0
                elif field_name in STANDARD_DTYPES:
                    default_dtype = STANDARD_DTYPES[field_name]
                    if 'u' in default_dtype or 'i' in default_dtype:
                        vertex_data[field_name] = 0
                    else:
                        vertex_data[field_name] = 0.0
                else:
                    vertex_data[field_name] = 0
        
        # Step 6: Write PLY file
        el = PlyElement.describe(vertex_data, 'vertex')
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        PlyData([el], text=False).write(output_path)
        
        # Summary
        fields_preserved = [f for f in field_order if f in data]
        return True, num_points, f"Standardized: {len(fields_preserved)} fields preserved"
        
    except Exception as e:
        return False, 0, f"Error: {str(e)}"


def process_directory(input_dir, output_dir, preserve_labels=False):
    """
    Process all files in input directory and save standardized versions.
    
    Args:
        input_dir: Input directory containing LAS/PLY files
        output_dir: Output directory for standardized PLY files
        preserve_labels: If True, preserve semantic_seg and treeID if present
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all supported files
    input_files = []
    input_files.extend(glob.glob(os.path.join(input_dir, "*.las")))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.laz")))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.ply")))
    input_files = sorted(input_files)
    
    if not input_files:
        print(f"⚠️  No LAS/LAZ/PLY files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} file(s) to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    successful = []
    failed = []
    
    for i, input_file in enumerate(input_files, 1):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.ply")
        
        print(f"[{i}/{len(input_files)}] Processing: {os.path.basename(input_file)}")
        
        success, num_points, message = standardize_file(
            input_file, 
            output_file,
            preserve_labels=preserve_labels
        )
        
        if success:
            print(f"  ✅ {message}")
            print(f"  Points: {num_points:,}")
            print(f"  Saved: {os.path.basename(output_file)}")
            successful.append((base_name, num_points))
        else:
            print(f"  ❌ Failed: {message}")
            failed.append((os.path.basename(input_file), message))
            # Remove failed output file if it exists
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful)}/{len(input_files)}")
    if successful:
        total_points = sum(points for _, points in successful)
        print(f"   Total points: {total_points:,}")
        for name, points in successful:
            print(f"   - {name}.ply: {points:,} points")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(input_files)}")
        for name, reason in failed:
            print(f"   - {name}: {reason}")
    
    print(f"\nOutput directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Standardize point cloud files to have consistent field names and data types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standardize files (coordinates + scanner features only)
  python tools/standardize_pointcloud_data.py --input-dir /path/to/input --output-dir /path/to/output
  
  # Standardize files and preserve labels
  python tools/standardize_pointcloud_data.py --input-dir /path/to/input --output-dir /path/to/output --preserve-labels
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing LAS/PLY files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for standardized PLY files'
    )
    parser.add_argument(
        '--preserve-labels',
        action='store_true',
        help='Preserve semantic_seg and treeID labels if present'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    process_directory(
        args.input_dir, 
        args.output_dir,
        preserve_labels=args.preserve_labels
    )


if __name__ == "__main__":
    main()

