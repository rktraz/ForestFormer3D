#!/usr/bin/env python3
"""
Prepare training/validation splits for three training scenarios:
1. Heidelberg-only dataset
2. UWaterloo-only dataset
3. Combined dataset (both Heidelberg + UWaterloo)

This script:
1. Creates train/val splits (80/20) for each scenario
2. Copies standardized PLY files to data/ForAINetV2/train_val_data/
3. Creates meta_data/train_list.txt and val_list.txt for each scenario
4. Prepares separate directories for each training scenario

Usage:
    python tools/prepare_training_splits.py --scenario heidelberg
    python tools/prepare_training_splits.py --scenario uwaterloo
    python tools/prepare_training_splits.py --scenario combined
    python tools/prepare_training_splits.py --scenario all  # Prepare all three
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
import glob


def get_standardized_files(dataset):
    """Get list of standardized PLY files for a dataset."""
    if dataset == 'heidelberg':
        data_dir = 'data/standardized/heidelberg'
    elif dataset == 'uwaterloo':
        data_dir = 'data/standardized/uwaterloo'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    ply_files = sorted(glob.glob(os.path.join(data_dir, '*.ply')))
    return [os.path.basename(f).replace('.ply', '') for f in ply_files]


def create_train_val_split(files, val_ratio=0.2, seed=42):
    """Create train/val split from list of files."""
    random.seed(seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    num_val = max(1, int(len(files_shuffled) * val_ratio))
    val_files = files_shuffled[:num_val]
    train_files = files_shuffled[num_val:]
    
    return train_files, val_files


def prepare_scenario(scenario, val_ratio=0.2, seed=42):
    """
    Prepare data for a training scenario.
    
    Args:
        scenario: 'heidelberg', 'uwaterloo', or 'combined'
        val_ratio: Validation split ratio (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for scenario: {scenario.upper()}")
    print(f"{'='*60}\n")
    
    # Get files based on scenario
    if scenario == 'heidelberg':
        files = get_standardized_files('heidelberg')
        source_dir = 'data/standardized/heidelberg'
    elif scenario == 'uwaterloo':
        files = get_standardized_files('uwaterloo')
        source_dir = 'data/standardized/uwaterloo'
    elif scenario == 'combined':
        heidelberg_files = get_standardized_files('heidelberg')
        uwaterloo_files = get_standardized_files('uwaterloo')
        files = heidelberg_files + uwaterloo_files
        source_dir = None  # Multiple source directories
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    print(f"Found {len(files)} file(s)")
    
    # Create train/val split
    train_files, val_files = create_train_val_split(files, val_ratio, seed)
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"\nTrain files: {train_files}")
    print(f"Val files: {val_files}")
    
    # Create output directory structure
    output_base = f'data/ForAINetV2_{scenario}'
    train_val_dir = os.path.join(output_base, 'train_val_data')
    meta_dir = os.path.join(output_base, 'meta_data')
    
    os.makedirs(train_val_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Copy PLY files to train_val_data
    print(f"\nCopying PLY files to {train_val_dir}...")
    
    if scenario == 'combined':
        # Copy from both directories
        heidelberg_source = 'data/standardized/heidelberg'
        uwaterloo_source = 'data/standardized/uwaterloo'
        
        for filename in train_files + val_files:
            # Try Heidelberg first
            source_file = os.path.join(heidelberg_source, f"{filename}.ply")
            if not os.path.exists(source_file):
                # Try UWaterloo
                source_file = os.path.join(uwaterloo_source, f"{filename}.ply")
            
            if os.path.exists(source_file):
                dest_file = os.path.join(train_val_dir, f"{filename}.ply")
                shutil.copy2(source_file, dest_file)
                print(f"  Copied: {filename}.ply")
            else:
                print(f"  ⚠️  Warning: File not found: {filename}.ply")
    else:
        # Copy from single source directory
        for filename in train_files + val_files:
            source_file = os.path.join(source_dir, f"{filename}.ply")
            dest_file = os.path.join(train_val_dir, f"{filename}.ply")
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"  Copied: {filename}.ply")
            else:
                print(f"  ⚠️  Warning: File not found: {filename}.ply")
    
    # Create train_list.txt and val_list.txt
    train_list_path = os.path.join(meta_dir, 'train_list.txt')
    val_list_path = os.path.join(meta_dir, 'val_list.txt')
    
    with open(train_list_path, 'w') as f:
        for filename in train_files:
            f.write(f"{filename}\n")
    
    with open(val_list_path, 'w') as f:
        for filename in val_files:
            f.write(f"{filename}\n")
    
    print(f"\n✅ Created train_list.txt: {len(train_files)} files")
    print(f"✅ Created val_list.txt: {len(val_files)} files")
    
    print(f"\n{'='*60}")
    print(f"✅ Scenario '{scenario}' prepared successfully!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_base}")
    print(f"Next steps:")
    print(f"  1. Copy batch_load script:")
    print(f"     cp data/ForAINetV2/batch_load_ForAINetV2_data.py {output_base}/")
    print(f"     cp data/ForAINetV2/load_forainetv2_data.py {output_base}/")
    print(f"  2. cd {output_base}")
    print(f"  3. python batch_load_ForAINetV2_data.py \\")
    print(f"       --train_scan_names_file meta_data/train_list.txt \\")
    print(f"       --val_scan_names_file meta_data/val_list.txt \\")
    print(f"       --test_scan_names_file meta_data/val_list.txt")
    print(f"  4. cd ../..")
    print(f"  5. python tools/create_data_forainetv2.py forainetv2 \\")
    print(f"       --root-path {output_base} \\")
    print(f"       --out-dir {output_base} \\")
    print(f"       --extra-tag forainetv2_{scenario}")
    print()
    
    return output_base, train_files, val_files


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training/validation splits for ForestFormer3D training scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Heidelberg-only scenario
  python tools/prepare_training_splits.py --scenario heidelberg
  
  # Prepare UWaterloo-only scenario
  python tools/prepare_training_splits.py --scenario uwaterloo
  
  # Prepare Combined scenario
  python tools/prepare_training_splits.py --scenario combined
  
  # Prepare all three scenarios
  python tools/prepare_training_splits.py --scenario all
        """
    )
    
    parser.add_argument(
        '--scenario',
        required=True,
        choices=['heidelberg', 'uwaterloo', 'combined', 'all'],
        help='Training scenario to prepare'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.scenario == 'all':
        # Prepare all three scenarios
        for scenario in ['heidelberg', 'uwaterloo', 'combined']:
            try:
                prepare_scenario(scenario, args.val_ratio, args.seed)
            except Exception as e:
                print(f"❌ Error preparing {scenario}: {e}")
                continue
    else:
        prepare_scenario(args.scenario, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()



