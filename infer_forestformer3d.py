#!/usr/bin/env python3
"""
ForestFormer3D Inference Script

Run ForestFormer3D inference on one or more point cloud files (PLY or LAS/LAZ).
This is a Python version of run_inference_single_file.sh with support for
multiple files and directories.

Usage:
    python infer_forestformer3d.py --input file1.ply file2.las
    python infer_forestformer3d.py --input directory/
    python infer_forestformer3d.py --input file.ply --output custom_output/
"""

import os
import sys
import glob
import shutil
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree

# Try to import GPU checking libraries
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

#####################################################################
# GPU DETECTION FUNCTIONS
#####################################################################

def get_gpu_memory_info():
    """
    Get GPU memory information for all available GPUs.
    
    Returns:
        list: List of dicts with keys: 'device_id', 'total_mb', 'used_mb', 'free_mb', 'utilization'
    """
    gpu_info = []
    
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info.append({
                    'device_id': i,
                    'total_mb': mem_info.total / (1024**2),
                    'used_mb': mem_info.used / (1024**2),
                    'free_mb': mem_info.free / (1024**2),
                    'utilization': util.gpu
                })
            
            pynvml.nvmlShutdown()
            return gpu_info
        except Exception as e:
            logger.debug(f"Failed to get GPU info via pynvml: {e}")
    
    # Fallback: parse nvidia-smi output
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpu_info.append({
                    'device_id': int(parts[0]),
                    'total_mb': float(parts[1]),
                    'used_mb': float(parts[2]),
                    'free_mb': float(parts[3]),
                    'utilization': float(parts[4])
                })
        
        return gpu_info
    except Exception as e:
        logger.debug(f"Failed to get GPU info via nvidia-smi: {e}")
        return []

def find_best_gpu(min_free_mb=5000, max_utilization=80):
    """
    Find the GPU with the most free memory that meets criteria.
    
    Args:
        min_free_mb: Minimum free memory in MB (default: 5GB)
        max_utilization: Maximum GPU utilization percentage (default: 80%)
        
    Returns:
        int or None: Device ID of best GPU, or None if none available
    """
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        logger.warning("Could not detect GPU information. Using default GPU 0.")
        return "0"
    
    # Filter GPUs that meet criteria
    suitable_gpus = [
        gpu for gpu in gpu_info
        if gpu['free_mb'] >= min_free_mb and gpu['utilization'] <= max_utilization
    ]
    
    if not suitable_gpus:
        return None
    
    # Sort by free memory (descending) and return the best one
    suitable_gpus.sort(key=lambda x: x['free_mb'], reverse=True)
    best_gpu = suitable_gpus[0]
    
    return str(best_gpu['device_id'])

def display_gpu_status():
    """Display status of all GPUs."""
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        logger.warning("Could not detect GPU information.")
        return
    
    logger.info("=" * 70)
    logger.info("GPU Status:")
    logger.info("=" * 70)
    logger.info(f"{'GPU':<6} {'Total (GB)':<12} {'Used (GB)':<12} {'Free (GB)':<12} {'Util %':<10}")
    logger.info("-" * 70)
    
    for gpu in gpu_info:
        total_gb = gpu['total_mb'] / 1024
        used_gb = gpu['used_mb'] / 1024
        free_gb = gpu['free_mb'] / 1024
        util = gpu['utilization']
        
        logger.info(f"{gpu['device_id']:<6} {total_gb:<12.2f} {used_gb:<12.2f} {free_gb:<12.2f} {util:<10.1f}")
    
    logger.info("=" * 70)

#####################################################################
# HELPER FUNCTIONS
#####################################################################

def rel_path(path):
    """
    Convert an absolute path to a relative path (relative to project root).
    If path is already relative or outside project root, return as-is.
    """
    if not path:
        return path
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.abspath(path)
        if abs_path.startswith(project_root):
            return os.path.relpath(abs_path, project_root)
        return path
    except:
        return path

#####################################################################
# CONFIGURATION
#####################################################################

class CONFIG:
    """Configuration constants for ForestFormer3D inference."""
    # Working directory
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Test data directory
    TEST_DATA_DIR = os.path.join(WORK_DIR, "data/ForAINetV2/test_data")
    META_DIR = os.path.join(WORK_DIR, "data/ForAINetV2/meta_data")
    
    # Model and config paths
    # Default to scanner features model (new training)
    CONFIG_FILE = os.path.join(WORK_DIR, "configs/oneformer3d_qs_radius16_qp300_2many_features.py")
    DEFAULT_MODEL_PATH = os.path.join(WORK_DIR, "work_dirs/heidelberg_features/epoch_500.pth")
    
    # Legacy model paths (for backward compatibility)
    LEGACY_CONFIG_FILE = os.path.join(WORK_DIR, "configs/oneformer3d_qs_radius16_qp300_2many.py")
    LEGACY_MODEL_PATH = os.path.join(WORK_DIR, "work_dirs/clean_forestformer/epoch_3000_fix.pth")
    
    # Default output directory (intermediate files)
    DEFAULT_INTERMEDIATE_OUTPUT_DIR = os.path.join(WORK_DIR, "work_dirs/output")
    
    # Final output directory (timestamped runs)
    INFERENCE_RUNS_DIR = os.path.join(WORK_DIR, "inference_runs")
    
    # Conversion script
    CONVERT_SCRIPT = os.path.join(WORK_DIR, "tools/convert_las_to_ply.py")
    
    # Conda environment setup
    CONDA_SH_PATH = "/home/rkaharly/miniconda/etc/profile.d/conda.sh"
    CONDA_ENV_NAME = "forestformer3d"

#####################################################################
# HELPER FUNCTIONS
#####################################################################

def find_model_checkpoint(custom_path=None):
    """Find the model checkpoint path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    # Try scanner features model first (new training)
    if os.path.exists(CONFIG.DEFAULT_MODEL_PATH):
        return CONFIG.DEFAULT_MODEL_PATH
    
    # Try to find best.pth or latest epoch in heidelberg_features
    heidelberg_dir = os.path.join(CONFIG.WORK_DIR, "work_dirs/heidelberg_features")
    if os.path.exists(heidelberg_dir):
        best_pth = os.path.join(heidelberg_dir, "best.pth")
        if os.path.exists(best_pth):
            return best_pth
        # Find latest epoch
        pth_files = glob.glob(os.path.join(heidelberg_dir, "epoch_*.pth"))
        if pth_files:
            # Sort by epoch number
            pth_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            return pth_files[-1]
    
    # Fallback to legacy model
    if os.path.exists(CONFIG.LEGACY_MODEL_PATH):
        logger.warning("Using legacy model. Consider training with scanner features.")
        return CONFIG.LEGACY_MODEL_PATH
    
    # Try to find any .pth file in work_dirs/clean_forestformer/
    work_dir = os.path.join(CONFIG.WORK_DIR, "work_dirs/clean_forestformer")
    if os.path.exists(work_dir):
        pth_files = glob.glob(os.path.join(work_dir, "*.pth"))
        if pth_files:
            # Prefer epoch_3000_fix.pth if exists
            preferred = [f for f in pth_files if 'epoch_3000_fix' in f]
            if preferred:
                return preferred[0]
            return pth_files[0]
    
    logger.warning(f"Could not find model checkpoint. Using default: {CONFIG.DEFAULT_MODEL_PATH}")
    return CONFIG.DEFAULT_MODEL_PATH

def collect_input_files(input_paths):
    """
    Collect all input files from paths (files or directories).
    
    Args:
        input_paths: List of file paths or directory paths
        
    Returns:
        List of valid input file paths
    """
    all_files = []
    
    for path in input_paths:
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}. Skipping.")
            continue
        
        if os.path.isdir(path):
            # If directory, find all supported files
            logger.info(f"Scanning directory: {path}")
            all_files.extend(glob.glob(os.path.join(path, "*.ply")))
            all_files.extend(glob.glob(os.path.join(path, "*.las")))
            all_files.extend(glob.glob(os.path.join(path, "*.laz")))
        else:
            # If file, add directly
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.ply', '.las', '.laz']:
                all_files.append(path)
            else:
                logger.warning(f"Unsupported file type: {path}. Skipping.")
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    
    return all_files

def convert_las_to_ply(input_file, output_file):
    """
    Convert LAS/LAZ file to PLY using the conversion script.
    
    Args:
        input_file: Path to input LAS/LAZ file
        output_file: Path to output PLY file
    """
    logger.info(f"Converting {os.path.basename(input_file)} to PLY format...")
    
    # Run the conversion script
    try:
        result = subprocess.run(
            [sys.executable, CONFIG.CONVERT_SCRIPT, input_file, output_file],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✅ Converted to: {rel_path(output_file)}")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def setup_conda_env():
    """Setup conda environment activation command."""
    # Return a command prefix that activates conda if needed
    if os.path.exists(CONFIG.CONDA_SH_PATH):
        return f"source {CONFIG.CONDA_SH_PATH} && conda activate {CONFIG.CONDA_ENV_NAME} && "
    return ""

def clean_temporary_data():
    """Clean temporary data directories."""
    dirs_to_clean = [
        os.path.join(CONFIG.WORK_DIR, "data/ForAINetV2/forainetv2_instance_data"),
        os.path.join(CONFIG.WORK_DIR, "data/ForAINetV2/semantic_mask"),
        os.path.join(CONFIG.WORK_DIR, "data/ForAINetV2/points"),
        os.path.join(CONFIG.WORK_DIR, "data/ForAINetV2/instance_mask"),
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            for file in glob.glob(os.path.join(dir_path, "*")):
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                    elif os.path.isdir(file):
                        shutil.rmtree(file)
                except Exception as e:
                    logger.debug(f"Could not remove {file}: {e}")

def generate_test_list(base_name):
    """Generate test_list.txt with the given base name."""
    os.makedirs(CONFIG.META_DIR, exist_ok=True)
    test_list = os.path.join(CONFIG.META_DIR, "test_list.txt")
    test_list_init = os.path.join(CONFIG.META_DIR, "test_list_initial.txt")
    
    with open(test_list, 'w') as f:
        f.write(f"{base_name}\n")
    
    shutil.copy(test_list, test_list_init)

def run_preprocessing():
    """Run data preprocessing steps."""
    data_dir = os.path.join(CONFIG.WORK_DIR, "data/ForAINetV2")
    test_list = os.path.join(CONFIG.META_DIR, "test_list.txt")
    
    try:
        result = subprocess.run(
            [sys.executable, "batch_load_ForAINetV2_data.py", 
             "--test_scan_names_file", "meta_data/test_list.txt"],
            cwd=data_dir,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preprocessing failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def create_data_info():
    """Create data info files."""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(CONFIG.WORK_DIR, "tools/create_data_forainetv2.py"), "forainetv2"],
            cwd=CONFIG.WORK_DIR,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data info creation failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def run_inference(config_file, model_path, cuda_device="0"):
    """Run ForestFormer3D inference."""
    env = os.environ.copy()
    env['PYTHONPATH'] = CONFIG.WORK_DIR + (f":{env.get('PYTHONPATH', '')}" if env.get('PYTHONPATH') else "")
    env['OMP_NUM_THREADS'] = '8'
    env['CUDA_VISIBLE_DEVICES'] = cuda_device
    env['CURRENT_ITERATION'] = '1'
    
    # Override data_root and ann_file to use standard location where test data is placed
    # This ensures compatibility with configs that use different data roots (e.g., heidelberg)
    standard_data_root = os.path.join(CONFIG.WORK_DIR, 'data/ForAINetV2')
    cfg_overrides = [
        f"test_dataloader.dataset.data_root='{standard_data_root}'",
        f"test_dataloader.dataset.ann_file='forainetv2_oneformer3d_infos_test.pkl'"
    ]
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(CONFIG.WORK_DIR, "tools/test.py"), config_file, model_path,
             "--cfg-options"] + cfg_overrides,
            cwd=CONFIG.WORK_DIR,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        # Log output even on success to help debug issues
        if result.stdout:
            logger.debug(f"test.py stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"test.py stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False

def copy_final_predictions(base_name, original_input_file, intermediate_output_dir, final_output_dir, tolerance=0.1):
    """
    Match ForestFormer3D prediction points to the original point cloud using spatial KDTree
    and save a new PLY with ALL original points, assigning predictions where available.
    
    The model processes a subset of points (after CylinderCrop, GridSample, PointSample),
    so predictions are only available for that subset. This function:
    1. Matches prediction points to original points
    2. Assigns predictions to the corresponding original points
    3. Keeps ALL original points (unmatched points get default/unknown predictions)
    
    Args:
        tolerance: Maximum distance (in meters) for matching points. Default 0.1m (10cm).
                   If matching fails with default tolerance, will try larger tolerances automatically.
    """
    # Look for the raw output file with predictions
    pred_file = os.path.join(intermediate_output_dir, f"{base_name}_1.ply")
    if not os.path.exists(pred_file):
        logger.warning(f"Output file not found at {pred_file}")
        logger.info("Checking for alternative output locations...")
        # Check for files with base_name
        alt_files = glob.glob(os.path.join(intermediate_output_dir, f"*{base_name}*"))
        if alt_files:
            logger.info(f"Found files matching '{base_name}': {alt_files[:5]}")
        # Check all recent PLY files
        all_ply_files = glob.glob(os.path.join(intermediate_output_dir, "*.ply"))
        if all_ply_files:
            # Sort by modification time, most recent first
            all_ply_files.sort(key=os.path.getmtime, reverse=True)
            logger.info(f"Most recent PLY files in output directory: {[os.path.basename(f) for f in all_ply_files[:5]]}")
        # Check subdirectories
        subdirs = [d for d in os.listdir(intermediate_output_dir) if os.path.isdir(os.path.join(intermediate_output_dir, d))]
        if subdirs:
            logger.info(f"Found subdirectories: {subdirs[:5]}")
            for subdir in subdirs[:3]:
                subdir_files = glob.glob(os.path.join(intermediate_output_dir, subdir, f"*{base_name}*.ply"))
                if subdir_files:
                    logger.info(f"  Found in {subdir}: {[os.path.basename(f) for f in subdir_files[:3]]}")
        logger.error(f"Could not find expected output file: {pred_file}")
        logger.error("This usually means the inference step failed or produced output with a different name.")
        return False

    # Load offsets (used by preprocessing) so we can center original coords to model space
    offset_file = os.path.join(
        CONFIG.WORK_DIR,
        "data/ForAINetV2/forainetv2_instance_data",
        f"{base_name}_offsets.npy",
    )
    offsets = None
    if os.path.exists(offset_file):
        try:
            import numpy as np

            offsets = np.load(offset_file)
            logger.info(f"Found coordinate offsets: {offsets}")
        except Exception as e:
            logger.warning(f"Could not load offsets: {e}")

    # Create final output filename
    final_filename = f"{base_name}_classified_by_ff3d.ply"
    final_output_path = os.path.join(final_output_dir, final_filename)

    try:
        from plyfile import PlyData, PlyElement
        import numpy as np

        # 1) Load original point cloud (world coordinates, all fields)
        file_ext = os.path.splitext(original_input_file)[1].lower()
        if file_ext in [".las", ".laz"]:
            # Use the converted PLY that was used for preprocessing
            original_ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
            if not os.path.exists(original_ply_file):
                logger.error(f"Converted PLY file not found: {original_ply_file}")
                return False
        else:
            original_ply_file = original_input_file

        original_ply = PlyData.read(original_ply_file)
        original_vertex = original_ply["vertex"].data
        orig_x = np.asarray(original_vertex["x"], dtype=np.float64)
        orig_y = np.asarray(original_vertex["y"], dtype=np.float64)
        orig_z = np.asarray(original_vertex["z"], dtype=np.float64)
        orig_points = np.stack([orig_x, orig_y, orig_z], axis=1)  # world coords

        # 2) Load prediction points (model coords, centered by offsets)
        pred_ply = PlyData.read(pred_file)
        pred_vertex = pred_ply["vertex"].data
        pred_x = np.asarray(pred_vertex["x"], dtype=np.float64)
        pred_y = np.asarray(pred_vertex["y"], dtype=np.float64)
        pred_z = np.asarray(pred_vertex["z"], dtype=np.float64)
        pred_points = np.stack([pred_x, pred_y, pred_z], axis=1)

        # 3) Build KDTree in the same coordinate system as predictions
        if offsets is not None:
            # Center original points using the same offsets that preprocessing used
            orig_centered = orig_points.copy()
            orig_centered[:, 0] -= offsets[0]
            orig_centered[:, 1] -= offsets[1]
            orig_centered[:, 2] -= offsets[2]
        else:
            # Fallback: assume predictions are already in world coordinates
            orig_centered = orig_points

        logger.info(f"  Building KDTree for {len(original_vertex)} original points...")
        tree = cKDTree(orig_centered)

        # 4) Match each prediction point to the nearest original point
        # Try with increasing tolerances if initial matching fails
        tolerances_to_try = [tolerance, tolerance * 2, tolerance * 5, tolerance * 10, 1.0, 5.0]
        matched_mask = None
        final_tolerance = tolerance
        distances = None
        indices = None
        
        for tol in tolerances_to_try:
            logger.info(f"  Matching {len(pred_vertex)} prediction points to {len(orig_points):,} original points (tolerance={tol:.3f}m)...")
            distances, indices = tree.query(pred_points, k=1)
            matched_mask = distances <= tol
            num_matched = int(np.sum(matched_mask))
            num_total = len(pred_vertex)
            
            match_rate = num_matched / num_total if num_total > 0 else 0
            logger.info(f"    Matched {num_matched:,}/{num_total:,} prediction points ({100*match_rate:.1f}%)")
            
            # If we get at least 50% match rate, use this tolerance
            if match_rate >= 0.5:
                final_tolerance = tol
                if tol > tolerance:
                    logger.warning(f"  Used larger tolerance ({tol:.3f}m) to achieve {100*match_rate:.1f}% match rate")
                break
        
        if matched_mask is None or num_matched == 0:
            logger.error("No prediction points matched to the original cloud even with large tolerance.")
            if distances is not None:
                logger.error(f"  Distance statistics: min={distances.min():.4f}m, max={distances.max():.4f}m, "
                            f"mean={distances.mean():.4f}m, median={np.median(distances):.4f}m")
            return False
        
        if num_matched < num_total * 0.5:
            logger.warning(f"  Only {100*num_matched/num_total:.1f}% of prediction points matched. "
                          f"This may indicate coordinate system misalignment.")

        # 5) Create arrays for ALL original points (not just matched ones)
        # Initialize prediction arrays for all original points with default values
        num_original = len(original_vertex)
        
        # Default values for unmatched points
        default_semantic = -1  # Unknown/unlabeled
        default_instance = -1  # Unknown/unlabeled
        default_score = 0.0
        
        # Create prediction arrays for all original points
        semantic_pred_all = np.full(num_original, default_semantic, dtype=np.int32)
        instance_pred_all = np.full(num_original, default_instance, dtype=np.int32)
        score_all = np.full(num_original, default_score, dtype=np.float32)
        
        # Assign predictions to matched original points
        matched_orig_indices = indices[matched_mask]
        semantic_pred_all[matched_orig_indices] = pred_vertex["semantic_pred"][matched_mask]
        if "instance_pred" in pred_vertex.dtype.names:
            instance_pred_all[matched_orig_indices] = pred_vertex["instance_pred"][matched_mask]
        if "score" in pred_vertex.dtype.names:
            score_all[matched_orig_indices] = pred_vertex["score"][matched_mask]
        
        logger.info(f"  Assigned predictions to {len(matched_orig_indices):,} original points")
        logger.info(f"  {num_original - len(matched_orig_indices):,} points have no predictions (will use default values)")

        # Create new dtype list starting with original fields
        dtype_list = list(original_vertex.dtype.descr)

        # Add prediction fields if not already present
        if "semantic_pred" not in original_vertex.dtype.names:
            dtype_list.append(("semantic_pred", "<i4"))
        if "instance_pred" not in original_vertex.dtype.names:
            dtype_list.append(("instance_pred", "<i4"))
        if "score" not in original_vertex.dtype.names:
            dtype_list.append(("score", "<f4"))

        # Create new structured array with ALL original data + predictions
        vertex_data = np.empty(num_original, dtype=dtype_list)

        # Copy ALL original fields
        for field_name in original_vertex.dtype.names:
            vertex_data[field_name] = original_vertex[field_name]

        # Add prediction fields for ALL points
        vertex_data["semantic_pred"] = semantic_pred_all
        vertex_data["instance_pred"] = instance_pred_all
        vertex_data["score"] = score_all

        # 6) Write PLY file (world coordinates, ALL original points)
        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el], text=False).write(final_output_path)

        logger.info(f"  Saved: {os.path.basename(final_output_path)}")
        logger.info(f"  Final output: {num_original:,} points (ALL original points preserved)")
        logger.info(f"    - {len(matched_orig_indices):,} points have predictions")
        logger.info(f"    - {num_original - len(matched_orig_indices):,} points have no predictions (default values)")
        return True
    except Exception as e:
        logger.error(f"Failed to process prediction file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_single_file(input_file, intermediate_output_dir, final_output_dir, model_path, config_file, cuda_device="0", keep_intermediate=False):
    """
    Process a single input file through the full inference pipeline.
    
    Args:
        input_file: Path to input file (PLY or LAS/LAZ)
        intermediate_output_dir: Output directory for intermediate results (work_dirs/output)
        final_output_dir: Final output directory (inference_runs/run_TIMESTAMP)
        model_path: Path to model checkpoint
        config_file: Path to config file
        cuda_device: CUDA device ID
        keep_intermediate: Whether to keep intermediate files
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get file extension and base name
    file_ext = os.path.splitext(input_file)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    logger.info(f"Processing: {os.path.basename(input_file)}")
    
    # Standardize file with caching (ensures consistent field names)
    try:
        from tools.standardize_with_cache import get_standardized_file
        standardized_file = get_standardized_file(
            input_file,
            preserve_labels=False,  # Don't preserve labels for inference
            force_recompute=False
        )
        if standardized_file is None:
            logger.warning(f"Standardization failed, using original file: {input_file}")
            standardized_file = input_file
        else:
            logger.info(f"  Using standardized file: {os.path.basename(standardized_file)}")
    except Exception as e:
        logger.warning(f"Standardization with cache failed ({e}), using original file")
        standardized_file = input_file
    
    # Copy standardized file to test_data directory
    os.makedirs(CONFIG.TEST_DATA_DIR, exist_ok=True)
    ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
    
    if standardized_file != input_file:
        # Use standardized version
        shutil.copy2(standardized_file, ply_file)
    elif file_ext in ['.las', '.laz']:
        # Convert LAS/LAZ to PLY (standardization should have handled this, but fallback)
        logger.info(f"  Converting to PLY...")
        if not convert_las_to_ply(input_file, ply_file):
            logger.error(f"Failed to convert {input_file}")
            return False
    elif file_ext == '.ply':
        # Copy PLY file as-is
        shutil.copy2(input_file, ply_file)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return False
    
    # Clean temporary data
    clean_temporary_data()
    
    # Generate test list
    generate_test_list(base_name)
    
    # Run preprocessing
    if not run_preprocessing():
        return False
    
    # Create data info
    if not create_data_info():
        return False
    
    # Run inference
    logger.info(f"  Running inference...")
    if not run_inference(config_file, model_path, cuda_device):
        return False
    
    # Save prediction file by matching predictions back to original point cloud
    if not copy_final_predictions(base_name, input_file, intermediate_output_dir, final_output_dir):
        logger.error(f"Failed to save final prediction for {base_name}")
        return False
    
    # Clean up intermediate files if requested
    if not keep_intermediate:
        cleanup_intermediate_files(base_name, intermediate_output_dir)
    
    return True

def cleanup_intermediate_files(base_name, intermediate_output_dir):
    """Remove intermediate files created during inference."""
    removed_count = 0
    
    # Remove files directly in output directory
    patterns_in_root = [
        f"{base_name}_1.ply",
        f"{base_name}_bluepoints_*.ply",
    ]
    
    for pattern in patterns_in_root:
        pattern_path = os.path.join(intermediate_output_dir, pattern)
        for file_path in glob.glob(pattern_path):
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                logger.debug(f"Could not remove {file_path}: {e}")
    
    # Remove files in subdirectories
    subdir_patterns = [
        ("round_*", f"{base_name}_*.ply"),
        ("round_*", f"{base_name}_*.las"),
        ("round_*_noisy_score", f"{base_name}_*.ply"),
        ("round_*_after_remove_noise_*", f"{base_name}_*.ply"),
    ]
    
    for subdir_pattern, file_pattern in subdir_patterns:
        subdir_glob = os.path.join(intermediate_output_dir, subdir_pattern)
        for subdir in glob.glob(subdir_glob):
            file_glob = os.path.join(subdir, file_pattern)
            for file_path in glob.glob(file_glob):
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.debug(f"Could not remove {file_path}: {e}")
    
    # Remove empty directories (do this after removing files)
    try:
        # Walk bottom-up to remove empty directories
        for root, dirs, files in os.walk(intermediate_output_dir, topdown=False):
            for dir_name in dirs:
                if dir_name.startswith('round_'):
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Check if directory is empty
                            os.rmdir(dir_path)
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        logger.debug(f"Could not remove empty directory {dir_path}: {e}")
    except Exception as e:
        logger.debug(f"Error during directory cleanup: {e}")
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} intermediate file(s)")

#####################################################################
# MAIN FUNCTION
#####################################################################

def run_inference_on_files(input_files, model_path, config_file=None, output_dir=None, cuda_device="0", keep_intermediate=False):
    """
    Run inference on multiple files programmatically.
    
    Args:
        input_files: List of input file paths (PLY or LAS/LAZ)
        model_path: Path to model checkpoint (REQUIRED)
        config_file: Path to config file (None = use default)
        output_dir: Output directory (None = create timestamped directory)
        cuda_device: CUDA device ID
        keep_intermediate: Whether to keep intermediate files
        
    Returns:
        str: Path to output directory containing predictions
    """
    # Change to work directory
    os.chdir(CONFIG.WORK_DIR)
    
    # Validate model checkpoint
    if model_path is None:
        raise ValueError("model_path is required. Please specify --model argument.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Get config file - auto-detect based on model
    if config_file is None:
        # Check if using scanner features model
        model_path_lower = model_path.lower()
        if 'features' in model_path_lower or 'heidelberg_features' in model_path_lower:
            # Use scenario-specific config if available, otherwise base config
            if 'heidelberg' in model_path_lower:
                heidelberg_config = os.path.join(CONFIG.WORK_DIR, "configs/oneformer3d_features_heidelberg.py")
                if os.path.exists(heidelberg_config):
                    config_file = heidelberg_config
                else:
                    config_file = CONFIG.CONFIG_FILE  # Base scanner features config
            elif 'uwaterloo' in model_path_lower:
                uwaterloo_config = os.path.join(CONFIG.WORK_DIR, "configs/oneformer3d_features_uwaterloo.py")
                if os.path.exists(uwaterloo_config):
                    config_file = uwaterloo_config
                else:
                    config_file = CONFIG.CONFIG_FILE
            elif 'combined' in model_path_lower:
                combined_config = os.path.join(CONFIG.WORK_DIR, "configs/oneformer3d_features_combined.py")
                if os.path.exists(combined_config):
                    config_file = combined_config
                else:
                    config_file = CONFIG.CONFIG_FILE
            else:
                config_file = CONFIG.CONFIG_FILE  # Base scanner features config
        else:
            config_file = CONFIG.LEGACY_CONFIG_FILE  # Legacy config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Get intermediate output directory (where model writes files)
    intermediate_output_dir = CONFIG.DEFAULT_INTERMEDIATE_OUTPUT_DIR
    os.makedirs(intermediate_output_dir, exist_ok=True)
    
    # Get final output directory (timestamped run directory)
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(CONFIG.INFERENCE_RUNS_DIR, f"run_{timestamp}")
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for i, input_file in enumerate(input_files, 1):
        if len(input_files) > 1:
            logger.info(f"[{i}/{len(input_files)}]")
        
        if not process_single_file(
            input_file, 
            intermediate_output_dir, 
            output_dir, 
            model_path, 
            config_file, 
            cuda_device,
            keep_intermediate=keep_intermediate
        ):
            raise RuntimeError(f"Inference failed for file: {input_file}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Run ForestFormer3D inference on point cloud files (PLY or LAS/LAZ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file (model required)
  python infer_forestformer3d.py --input file.ply --model work_dirs/heidelberg_features/epoch_500.pth
  
  # Process multiple files
  python infer_forestformer3d.py --input file1.ply file2.las file3.ply --model work_dirs/heidelberg_features/epoch_500.pth
  
  # Process all files in a directory
  python infer_forestformer3d.py --input directory/ --model work_dirs/heidelberg_features/epoch_500.pth
  
  # Custom output directory
  python infer_forestformer3d.py --input file.ply --model work_dirs/heidelberg_features/epoch_500.pth --output custom_output/
        """
    )
    
    parser.add_argument(
        "--input", 
        nargs='+',
        required=True,
        help="One or more input files or directories containing point cloud files (.ply, .las, .laz)"
    )
    parser.add_argument(
        "--output",
        help=f"Final output directory (default: inference_runs/run_TIMESTAMP)"
    )
    parser.add_argument(
        "--keep-intermediate-files",
        action="store_true",
        help="Keep intermediate files in work_dirs/output (default: False, files are cleaned up)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint .pth file (REQUIRED)"
    )
    parser.add_argument(
        "--config",
        help=f"Path to config file (default: {CONFIG.CONFIG_FILE})"
    )
    parser.add_argument(
        "--cuda-device",
        default="auto",
        help="CUDA device ID (default: 'auto' to auto-select, or specify 0, 1, 2, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress logs"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Change to work directory
    os.chdir(CONFIG.WORK_DIR)
    
    # Collect input files
    input_files = collect_input_files(args.input)
    
    if not input_files:
        logger.error("No valid input files found. Exiting.")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} file(s) to process:")
    for f in input_files:
        logger.info(f"  - {f}")
    
    # Determine output directory for display
    if args.output:
        display_output = os.path.abspath(args.output)
    else:
        display_output = "inference_runs/run_TIMESTAMP"
    
    # Auto-select GPU if requested
    cuda_device = args.cuda_device
    if cuda_device.lower() == "auto":
        logger.info("Auto-detecting best available GPU...")
        best_gpu = find_best_gpu(min_free_mb=5000, max_utilization=80)
        
        if best_gpu is None:
            logger.error("No suitable GPU found (need at least 5GB free memory and <80% utilization)")
            display_gpu_status()
            logger.error("\nAll GPUs are busy or don't have enough free memory.")
            logger.error("Please free up GPU memory or specify a GPU manually with --cuda-device")
            sys.exit(1)
        
        cuda_device = best_gpu
        gpu_info = get_gpu_memory_info()
        selected_gpu = next((g for g in gpu_info if str(g['device_id']) == cuda_device), None)
        if selected_gpu:
            logger.info(f"Selected GPU {cuda_device}: {selected_gpu['free_mb']/1024:.2f} GB free, "
                       f"{selected_gpu['utilization']:.1f}% utilization")
        else:
            logger.info(f"Selected GPU {cuda_device}")
    else:
        # Validate manually specified GPU
        try:
            device_id = int(cuda_device)
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                gpu = next((g for g in gpu_info if g['device_id'] == device_id), None)
                if gpu:
                    logger.info(f"Using GPU {device_id}: {gpu['free_mb']/1024:.2f} GB free, "
                               f"{gpu['utilization']:.1f}% utilization")
                else:
                    logger.warning(f"GPU {device_id} not found in system")
        except ValueError:
            logger.warning(f"Invalid GPU device ID: {cuda_device}, using as-is")
    
    logger.info(f"Using model checkpoint: {find_model_checkpoint(args.model)}")
    logger.info(f"Output: {display_output}")
    logger.info("")
    
    try:
        # Run inference
        output_dir = run_inference_on_files(
            input_files=input_files,
            model_path=args.model,
            config_file=args.config,
            output_dir=args.output,
            cuda_device=cuda_device,
            keep_intermediate=args.keep_intermediate_files
        )
        
        logger.info("")
        logger.info(f"✅ Inference complete. Results: {rel_path(output_dir)}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

