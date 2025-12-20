#!/usr/bin/env python3
"""
Inspect the contents of a .pkl annotation file to debug data loading issues.

Usage:
    python tools/inspect_pkl.py data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl
"""

import pickle
import sys
import os
from pathlib import Path

def inspect_pkl(pkl_path):
    """Load and display contents of a .pkl file."""
    print(f"Loading: {pkl_path}")
    print("=" * 80)
    
    if not os.path.exists(pkl_path):
        print(f"❌ Error: File not found: {pkl_path}")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Successfully loaded {pkl_path}")
        print(f"\nType: {type(data)}")
        
        if isinstance(data, list):
            print(f"\n📊 Total entries: {len(data)}")
            
            if len(data) > 0:
                print(f"\n📋 First entry structure:")
                print(f"   Type: {type(data[0])}")
                
                if isinstance(data[0], dict):
                    print(f"\n   Keys: {list(data[0].keys())}")
                    
                    # Show first entry in detail
                    first_entry = data[0]
                    print(f"\n   First entry details:")
                    for key, value in first_entry.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            print(f"     {key}: {value}")
                        elif isinstance(value, (list, tuple)):
                            print(f"     {key}: {type(value).__name__} of length {len(value)}")
                            if len(value) > 0:
                                print(f"            First element: {value[0]}")
                        elif isinstance(value, dict):
                            print(f"     {key}: dict with keys {list(value.keys())}")
                        else:
                            print(f"     {key}: {type(value).__name__}")
                    
                    # Check for file paths
                    print(f"\n   📁 File path checks:")
                    data_root = os.path.dirname(os.path.dirname(pkl_path))  # Go up from ForAINetV2
                    if 'pts_path' in first_entry:
                        pts_path = first_entry['pts_path']
                        full_path = os.path.join(data_root, pts_path) if not os.path.isabs(pts_path) else pts_path
                        exists = os.path.exists(full_path)
                        print(f"     pts_path: {pts_path}")
                        print(f"     Full path: {full_path}")
                        print(f"     Exists: {'✅' if exists else '❌'}")
                    
                    if 'pts_instance_mask_path' in first_entry:
                        mask_path = first_entry['pts_instance_mask_path']
                        full_path = os.path.join(data_root, mask_path) if not os.path.isabs(mask_path) else mask_path
                        exists = os.path.exists(full_path)
                        print(f"     pts_instance_mask_path: {mask_path}")
                        print(f"     Full path: {full_path}")
                        print(f"     Exists: {'✅' if exists else '❌'}")
                    
                    if 'pts_semantic_mask_path' in first_entry:
                        sem_path = first_entry['pts_semantic_mask_path']
                        full_path = os.path.join(data_root, sem_path) if not os.path.isabs(sem_path) else sem_path
                        exists = os.path.exists(full_path)
                        print(f"     pts_semantic_mask_path: {sem_path}")
                        print(f"     Full path: {full_path}")
                        print(f"     Exists: {'✅' if exists else '❌'}")
                
                # Show a few more entries
                if len(data) > 1:
                    print(f"\n   Sample of other entries (showing file names):")
                    for i in range(1, min(4, len(data))):
                        if isinstance(data[i], dict) and 'pts_path' in data[i]:
                            print(f"     Entry {i}: {data[i].get('pts_path', 'N/A')}")
        
        elif isinstance(data, dict):
            print(f"\n📊 Dictionary with keys: {list(data.keys())}")
            for key, value in data.items():
                if key == 'data_list' and isinstance(value, list):
                    print(f"   {key}: list of length {len(value)}")
                    if len(value) > 0:
                        print(f"\n   First entry in data_list:")
                        first_entry = value[0]
                        if isinstance(first_entry, dict):
                            print(f"     Keys: {list(first_entry.keys())}")
                            for k, v in first_entry.items():
                                if isinstance(v, (str, int, float, bool, type(None))):
                                    print(f"       {k}: {v}")
                                elif isinstance(v, (list, tuple)):
                                    print(f"       {k}: {type(v).__name__} of length {len(v)}")
                                else:
                                    print(f"       {k}: {type(v).__name__}")
                            
                            # Check file paths
                            print(f"\n     📁 File path checks:")
                            data_root = os.path.dirname(os.path.dirname(pkl_path))  # Go up from ForAINetV2
                            if 'pts_path' in first_entry:
                                pts_path = first_entry['pts_path']
                                full_path = os.path.join(data_root, pts_path) if not os.path.isabs(pts_path) else pts_path
                                exists = os.path.exists(full_path)
                                print(f"       pts_path: {pts_path}")
                                print(f"       Full path: {full_path}")
                                print(f"       Exists: {'✅' if exists else '❌'}")
                            
                            if 'pts_instance_mask_path' in first_entry:
                                mask_path = first_entry['pts_instance_mask_path']
                                full_path = os.path.join(data_root, mask_path) if not os.path.isabs(mask_path) else mask_path
                                exists = os.path.exists(full_path)
                                print(f"       pts_instance_mask_path: {mask_path}")
                                print(f"       Full path: {full_path}")
                                print(f"       Exists: {'✅' if exists else '❌'}")
                            
                            if 'pts_semantic_mask_path' in first_entry:
                                sem_path = first_entry['pts_semantic_mask_path']
                                full_path = os.path.join(data_root, sem_path) if not os.path.isabs(sem_path) else sem_path
                                exists = os.path.exists(full_path)
                                print(f"       pts_semantic_mask_path: {sem_path}")
                                print(f"       Full path: {full_path}")
                                print(f"       Exists: {'✅' if exists else '❌'}")
                elif key == 'metainfo' and isinstance(value, dict):
                    print(f"   {key}: dict with keys {list(value.keys())}")
                    for k, v in value.items():
                        print(f"     {k}: {v}")
                else:
                    print(f"   {key}: {type(value).__name__}")
        
        else:
            print(f"\n📊 Data structure: {type(data)}")
            print(f"   Preview: {str(data)[:200]}...")
    
    except Exception as e:
        print(f"❌ Error loading .pkl file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_pkl.py <path_to_pkl_file>")
        print("\nExample:")
        print("  python tools/inspect_pkl.py data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    inspect_pkl(pkl_path)

