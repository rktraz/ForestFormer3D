#!/usr/bin/env python3
"""
Convert LAS/PLY files to ForestFormer3D training format.

This script:
1. Converts LAS/LAZ to PLY if needed (using convert_las_to_ply.py)
2. Maps classification labels: 0→2 (wood), 1→3 (leaf)
3. Sets treeID = 1 for all points (one tree per file)
4. Removes scanner features, keeps only: x, y, z, semantic_seg, treeID
5. Saves to output directory following ForAINetV2 structure

Usage:
    python tools/prepare_training_data.py --input-dir /path/to/input --output-dir /path/to/output
    python tools/prepare_training_data.py --input-dir test_input_dir --output-dir test_output_dir
"""

import os
import sys
import glob
import argparse
import subprocess
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def convert_las_to_ply_temp(input_path, temp_ply_path):
    """Convert LAS/LAZ to PLY using the existing conversion script."""
    script_path = os.path.join(os.path.dirname(__file__), "convert_las_to_ply.py")
    result = subprocess.run(
        [sys.executable, script_path, input_path, temp_ply_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"LAS conversion failed: {result.stderr}")
    return temp_ply_path


def read_ply_file(ply_path):
    """Read PLY file and return data dictionary."""
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data
    
    # Convert to dictionary for easier access
    data = {}
    for name in vertex_data.dtype.names:
        data[name] = np.asarray(vertex_data[name])
    
    return data, len(vertex_data)


def convert_file_to_training_format(input_path, output_path):
    """
    Convert a single file to training format.
    
    Args:
        input_path: Path to input file (LAS/LAZ/PLY)
        output_path: Path to output PLY file
        
    Returns:
        tuple: (success: bool, num_points: int, message: str)
    """
    input_ext = os.path.splitext(input_path)[1].lower()
    temp_ply = None
    
    try:
        # Step 1: Convert LAS/LAZ to PLY if needed
        if input_ext in ['.las', '.laz']:
            temp_ply = output_path + '.temp.ply'
            convert_las_to_ply_temp(input_path, temp_ply)
            ply_path = temp_ply
        elif input_ext == '.ply':
            ply_path = input_path
        else:
            return False, 0, f"Unsupported file format: {input_ext}"
        
        # Step 2: Read PLY file
        data, num_points = read_ply_file(ply_path)
        
        # Step 3: Extract coordinates
        if 'x' not in data or 'y' not in data or 'z' not in data:
            return False, 0, "Missing x, y, or z coordinates"
        
        x = data['x'].astype(np.float64)
        y = data['y'].astype(np.float64)
        z = data['z'].astype(np.float64)
        
        # Step 4: Map classification labels
        # Find classification field (could be 'classification' or 'scalar_Classification' or similar)
        classification = None
        for field_name in ['scalar_Classification', 'classification', 'scalar_classification', 
                          'Classification', 'class', 'scalar_Class']:
            if field_name in data:
                classification = data[field_name]
                break
        
        if classification is None:
            return False, 0, f"Could not find classification field. Available fields: {list(data.keys())}"
        
        # Convert to int for comparison (handles float values like 0.0, 1.0)
        classification_int = classification.astype(np.int64)
        
        # Map: 0 (wood) → 2, 1 (leaf) → 3
        # semantic_seg should be 1-indexed: 1=ground, 2=wood, 3=leaf
        semantic_seg = np.zeros_like(classification_int, dtype=np.int64)
        wood_mask = (classification_int == 0)
        leaf_mask = (classification_int == 1)
        semantic_seg[wood_mask] = 2  # wood
        semantic_seg[leaf_mask] = 3  # leaf
        
        # Check if all points were mapped
        unmapped = np.sum((semantic_seg == 0) & (classification_int != 0) & (classification_int != 1))
        if unmapped > 0:
            unique_vals = np.unique(classification_int)
            return False, 0, f"Found unmapped classification values: {unique_vals}. Expected only 0 (wood) and 1 (leaf)."
        
        # Step 5: Set treeID = 1 for all points
        treeID = np.ones(num_points, dtype=np.int64)
        
        # Step 6: Create new PLY with only required fields
        dtype_list = [
            ('x', 'f8'),      # float64
            ('y', 'f8'),
            ('z', 'f8'),
            ('semantic_seg', 'i4'),  # int32
            ('treeID', 'i4')
        ]
        
        vertex_data = np.empty(num_points, dtype=dtype_list)
        vertex_data['x'] = x
        vertex_data['y'] = y
        vertex_data['z'] = z
        vertex_data['semantic_seg'] = semantic_seg
        vertex_data['treeID'] = treeID
        
        # Step 7: Write binary PLY file
        el = PlyElement.describe(vertex_data, 'vertex')
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        PlyData([el], text=False).write(output_path)
        
        # Cleanup temp file
        if temp_ply and os.path.exists(temp_ply):
            os.remove(temp_ply)
        
        # Count wood and leaf points
        num_wood = np.sum(wood_mask)
        num_leaf = np.sum(leaf_mask)
        
        return True, num_points, f"Converted: {num_wood} wood, {num_leaf} leaf points"
        
    except Exception as e:
        # Cleanup temp file on error
        if temp_ply and os.path.exists(temp_ply):
            try:
                os.remove(temp_ply)
            except:
                pass
        return False, 0, f"Error: {str(e)}"


def process_directory(input_dir, output_dir, output_subdir='train_val_data'):
    """
    Process all files in input directory and save to output directory.
    
    Args:
        input_dir: Input directory containing LAS/PLY files
        output_dir: Base output directory
        output_subdir: Subdirectory name (train_val_data or test_data)
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    output_data_dir = os.path.join(output_dir, output_subdir)
    
    os.makedirs(output_data_dir, exist_ok=True)
    
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
    print(f"Output directory: {output_data_dir}")
    print("-" * 60)
    
    successful = []
    failed = []
    
    for i, input_file in enumerate(input_files, 1):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_data_dir, f"{base_name}.ply")
        
        print(f"[{i}/{len(input_files)}] Processing: {os.path.basename(input_file)}")
        
        success, num_points, message = convert_file_to_training_format(input_file, output_file)
        
        if success:
            print(f"  ✅ {message}")
            print(f"  Saved: {os.path.relpath(output_file, output_dir)}")
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
    
    print(f"\nOutput directory: {output_data_dir}")
    print(f"Next steps:")
    print(f"  1. Create train_list.txt and val_list.txt in {os.path.join(output_dir, 'meta_data')}")
    print(f"  2. Run data preprocessing: cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py")


def verify_output_file(ply_path):
    """
    Verify that output PLY file has correct format and fields.
    
    Returns:
        tuple: (is_valid: bool, issues: list, info: dict)
    """
    issues = []
    info = {}
    
    try:
        # Read the file
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        
        # Check required fields
        required_fields = ['x', 'y', 'z', 'semantic_seg', 'treeID']
        available_fields = list(vertex_data.dtype.names)
        
        missing_fields = [f for f in required_fields if f not in available_fields]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Check field types (accounting for endianness: <f8, >f8, =f8 are all valid)
        field_types = {
            'x': ['f8', '<f8', '>f8', '=f8'],
            'y': ['f8', '<f8', '>f8', '=f8'],
            'z': ['f8', '<f8', '>f8', '=f8'],
            'semantic_seg': ['i4', '<i4', '>i4', '=i4'],
            'treeID': ['i4', '<i4', '>i4', '=i4']
        }
        
        for field, expected_types in field_types.items():
            if field in available_fields:
                actual_type = vertex_data.dtype[field].str
                # Remove endianness prefix for comparison
                actual_base = actual_type.lstrip('<>=') if len(actual_type) > 2 else actual_type
                expected_base = expected_types[0]
                if actual_base != expected_base:
                    issues.append(f"Field '{field}' has type {actual_type}, expected one of {expected_types}")
        
        # Check for extra fields (should only have the 5 required)
        extra_fields = [f for f in available_fields if f not in required_fields]
        if extra_fields:
            issues.append(f"Unexpected extra fields: {extra_fields}")
        
        # Check semantic_seg values (should be 2 or 3)
        if 'semantic_seg' in available_fields:
            semantic_seg = np.asarray(vertex_data['semantic_seg'])
            unique_vals = np.unique(semantic_seg)
            invalid_vals = [v for v in unique_vals if v not in [2, 3]]
            if invalid_vals:
                issues.append(f"semantic_seg contains invalid values: {invalid_vals}. Expected only 2 (wood) or 3 (leaf)")
            
            num_wood = np.sum(semantic_seg == 2)
            num_leaf = np.sum(semantic_seg == 3)
            info['num_wood'] = int(num_wood)
            info['num_leaf'] = int(num_leaf)
        
        # Check treeID values (should all be 1)
        if 'treeID' in available_fields:
            treeID = np.asarray(vertex_data['treeID'])
            unique_treeids = np.unique(treeID)
            if len(unique_treeids) != 1 or unique_treeids[0] != 1:
                issues.append(f"treeID should all be 1, but found: {unique_treeids}")
            info['treeID_value'] = int(unique_treeids[0])
        
        # Check file format (should be binary)
        with open(ply_path, 'rb') as f:
            header = f.read(200).decode('ascii', errors='ignore')
            if 'format ascii' in header:
                issues.append("File is ASCII format, should be binary")
            elif 'format binary' not in header:
                issues.append("Could not determine file format")
        
        info['num_points'] = len(vertex_data)
        info['fields'] = available_fields
        info['file_size_mb'] = os.path.getsize(ply_path) / (1024 * 1024)
        
        is_valid = len(issues) == 0
        
    except Exception as e:
        issues.append(f"Error reading file: {str(e)}")
        is_valid = False
    
    return is_valid, issues, info


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAS/PLY files to ForestFormer3D training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files for training
  python tools/prepare_training_data.py --input-dir /path/to/data --output-dir data/ForAINetV2
  
  # Process test files
  python tools/prepare_training_data.py --input-dir test_input_dir --output-dir test_output_dir --subdir test_data
  
  # Verify output files
  python tools/prepare_training_data.py --input-dir test_input_dir --output-dir test_output_dir --verify-only
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
        help='Output directory (will create train_val_data/ or test_data/ subdirectory)'
    )
    parser.add_argument(
        '--subdir',
        default='train_val_data',
        choices=['train_val_data', 'test_data'],
        help='Output subdirectory name (default: train_val_data)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing output files, do not convert'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if args.verify_only:
        # Verify existing files
        output_data_dir = os.path.join(args.output_dir, args.subdir)
        if not os.path.exists(output_data_dir):
            print(f"Error: Output directory does not exist: {output_data_dir}")
            sys.exit(1)
        
        ply_files = glob.glob(os.path.join(output_data_dir, "*.ply"))
        if not ply_files:
            print(f"No PLY files found in {output_data_dir}")
            sys.exit(1)
        
        print(f"Verifying {len(ply_files)} file(s) in {output_data_dir}")
        print("=" * 60)
        
        all_valid = True
        for ply_file in sorted(ply_files):
            base_name = os.path.basename(ply_file)
            is_valid, issues, info = verify_output_file(ply_file)
            
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"{status}: {base_name}")
            print(f"  Points: {info.get('num_points', 'N/A'):,}")
            if 'num_wood' in info:
                print(f"  Wood: {info['num_wood']:,}, Leaf: {info['num_leaf']:,}")
            if 'treeID_value' in info:
                print(f"  treeID: {info['treeID_value']}")
            print(f"  Size: {info.get('file_size_mb', 0):.2f} MB")
            print(f"  Fields: {', '.join(info.get('fields', []))}")
            
            if issues:
                print(f"  Issues:")
                for issue in issues:
                    print(f"    - {issue}")
                all_valid = False
            print()
        
        if all_valid:
            print("✅ All files are valid!")
        else:
            print("❌ Some files have issues. Please fix and retry.")
            sys.exit(1)
    else:
        # Process files
        process_directory(args.input_dir, args.output_dir, args.subdir)
        
        # Auto-verify after conversion
        print("\n" + "=" * 60)
        print("AUTO-VERIFICATION")
        print("=" * 60)
        output_data_dir = os.path.join(args.output_dir, args.subdir)
        ply_files = glob.glob(os.path.join(output_data_dir, "*.ply"))
        
        if ply_files:
            all_valid = True
            for ply_file in sorted(ply_files):
                base_name = os.path.basename(ply_file)
                is_valid, issues, info = verify_output_file(ply_file)
                
                if not is_valid:
                    all_valid = False
                    print(f"❌ {base_name}:")
                    for issue in issues:
                        print(f"   - {issue}")
                    # Remove invalid file
                    try:
                        os.remove(ply_file)
                        print(f"   Removed invalid file")
                    except:
                        pass
                else:
                    print(f"✅ {base_name}: {info.get('num_points', 0):,} points, "
                          f"Wood: {info.get('num_wood', 0):,}, Leaf: {info.get('num_leaf', 0):,}")
            
            if all_valid:
                print("\n✅ All converted files passed verification!")
            else:
                print("\n❌ Some files failed verification and were removed.")


if __name__ == "__main__":
    main()

