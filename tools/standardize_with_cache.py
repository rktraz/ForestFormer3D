#!/usr/bin/env python3
"""
Standardization with caching utility.

Provides transparent standardization of point cloud files with automatic caching.
Uses file hash for cache key to ensure cache invalidation on file changes.
"""

import os
import sys
import hashlib
import logging
from pathlib import Path

# Import standardization function
# Add tools directory to path to import standardize_pointcloud_data
tools_dir = os.path.dirname(os.path.abspath(__file__))
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)
from standardize_pointcloud_data import standardize_file

logger = logging.getLogger(__name__)


def compute_file_hash(file_path, chunk_size=8192):
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Chunk size for reading (default: 8KB)
        
    Returns:
        str: Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return None


def get_cache_key(input_path):
    """
    Generate cache key from file path and hash.
    
    Args:
        input_path: Path to input file
        
    Returns:
        tuple: (cache_filename, file_hash) or (None, None) on error
    """
    if not os.path.exists(input_path):
        return None, None
    
    # Compute file hash
    file_hash = compute_file_hash(input_path)
    if file_hash is None:
        return None, None
    
    # Get base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create cache filename: {base_name}_{hash}.ply
    cache_filename = f"{base_name}_{file_hash[:16]}.ply"  # Use first 16 chars of hash
    
    return cache_filename, file_hash


def get_standardized_file(input_path, cache_dir=None, preserve_labels=False, force_recompute=False):
    """
    Get standardized version of a point cloud file, using cache if available.
    
    This function:
    1. Computes file hash to generate cache key
    2. Checks if cached standardized version exists
    3. If cache exists and file unchanged, returns cached path
    4. If cache missing or file changed, standardizes and caches result
    5. Returns path to standardized PLY file
    
    Args:
        input_path: Path to input file (PLY, LAS, or LAZ)
        cache_dir: Cache directory (default: work_dirs/standardized_cache/)
        preserve_labels: If True, preserve semantic_seg and treeID labels
        force_recompute: If True, force re-standardization even if cache exists
        
    Returns:
        str: Path to standardized PLY file, or None on error
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return None
    
    # Set default cache directory
    if cache_dir is None:
        # Use work_dirs/standardized_cache/ relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "work_dirs", "standardized_cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key
    cache_filename, file_hash = get_cache_key(input_path)
    if cache_filename is None:
        logger.error(f"Failed to generate cache key for {input_path}")
        return None
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache exists and is valid
    if not force_recompute and os.path.exists(cache_path):
        # Verify source file hasn't changed by recomputing hash
        current_hash = compute_file_hash(input_path)
        if current_hash == file_hash:
            logger.info(f"Using cached standardized file: {os.path.basename(cache_path)}")
            return cache_path
        else:
            logger.info(f"Source file changed, re-standardizing: {os.path.basename(input_path)}")
            # Remove old cache
            try:
                os.remove(cache_path)
            except:
                pass
    
    # Standardize file
    logger.info(f"Standardizing file: {os.path.basename(input_path)}")
    success, num_points, message = standardize_file(
        input_path,
        cache_path,
        preserve_labels=preserve_labels
    )
    
    if success:
        logger.info(f"  ✅ {message} ({num_points:,} points)")
        logger.info(f"  Cached: {os.path.basename(cache_path)}")
        return cache_path
    else:
        logger.error(f"  ❌ Standardization failed: {message}")
        return None


def standardize_files_batch(input_files, cache_dir=None, preserve_labels=False, force_recompute=False):
    """
    Standardize multiple files with caching.
    
    Args:
        input_files: List of input file paths
        cache_dir: Cache directory (default: work_dirs/standardized_cache/)
        preserve_labels: If True, preserve semantic_seg and treeID labels
        force_recompute: If True, force re-standardization even if cache exists
        
    Returns:
        dict: Mapping from input_path -> standardized_path (or None if failed)
    """
    results = {}
    
    for input_path in input_files:
        standardized_path = get_standardized_file(
            input_path,
            cache_dir=cache_dir,
            preserve_labels=preserve_labels,
            force_recompute=force_recompute
        )
        results[input_path] = standardized_path
    
    return results


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize file with caching")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--preserve-labels", action="store_true", help="Preserve labels")
    parser.add_argument("--force", action="store_true", help="Force re-standardization")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = get_standardized_file(
        args.input,
        cache_dir=args.cache_dir,
        preserve_labels=args.preserve_labels,
        force_recompute=args.force
    )
    
    if result:
        print(f"Standardized file: {result}")
    else:
        print("Standardization failed")
        sys.exit(1)

