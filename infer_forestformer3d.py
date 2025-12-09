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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    CONFIG_FILE = os.path.join(WORK_DIR, "configs/oneformer3d_qs_radius16_qp300_2many.py")
    DEFAULT_MODEL_PATH = os.path.join(WORK_DIR, "work_dirs/clean_forestformer/epoch_3000_fix.pth")
    
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
    
    if os.path.exists(CONFIG.DEFAULT_MODEL_PATH):
        return CONFIG.DEFAULT_MODEL_PATH
    
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
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(CONFIG.WORK_DIR, "tools/test.py"), config_file, model_path],
            cwd=CONFIG.WORK_DIR,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def copy_final_predictions(base_name, original_input_file, intermediate_output_dir, final_output_dir, tolerance=0.01):
    """
    Match ForestFormer3D prediction points to the original point cloud using spatial KDTree
    and save a new PLY containing ONLY matched points, with original coordinates and
    all original fields plus semantic_pred / instance_pred / score.
    
    Unmatched original points are dropped.
    """
    # Look for the raw output file with predictions
    pred_file = os.path.join(intermediate_output_dir, f"{base_name}_1.ply")
    if not os.path.exists(pred_file):
        logger.warning(f"Output file not found at {pred_file}")
        logger.info("Checking for alternative output locations...")
        alt_files = glob.glob(os.path.join(intermediate_output_dir, f"*{base_name}*"))
        if alt_files:
            logger.info(f"Found: {alt_files[:5]}")
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
        logger.info(f"  Matching {len(pred_vertex)} prediction points (tolerance={tolerance}m)...")
        distances, indices = tree.query(pred_points, k=1)
        matched_mask = distances <= tolerance
        num_matched = int(np.sum(matched_mask))
        num_total = len(pred_vertex)

        if num_matched == 0:
            logger.error("No prediction points matched to the original cloud within tolerance.")
            return False

        logger.info(f"  Matched {num_matched}/{num_total} prediction points to original cloud")

        # 5) Build new vertex array using ONLY matched original points
        matched_orig_indices = indices[matched_mask]

        # Extract prediction fields (filtered to matched points)
        semantic_pred = pred_vertex["semantic_pred"][matched_mask]
        instance_pred = (
            pred_vertex["instance_pred"][matched_mask]
            if "instance_pred" in pred_vertex.dtype.names
            else None
        )
        score = (
            pred_vertex["score"][matched_mask]
            if "score" in pred_vertex.dtype.names
            else None
        )

        # Create new dtype list starting with original fields
        dtype_list = list(original_vertex.dtype.descr)

        # Add prediction fields if not already present
        if "semantic_pred" not in original_vertex.dtype.names:
            dtype_list.append(("semantic_pred", "<i4"))
        if instance_pred is not None and "instance_pred" not in original_vertex.dtype.names:
            dtype_list.append(("instance_pred", "<i4"))
        if score is not None and "score" not in original_vertex.dtype.names:
            dtype_list.append(("score", "<f4"))

        # Create new structured array with matched original data + predictions
        vertex_data = np.empty(num_matched, dtype=dtype_list)

        # Copy original fields for matched points
        for field_name in original_vertex.dtype.names:
            vertex_data[field_name] = original_vertex[field_name][matched_orig_indices]

        # Add prediction fields
        vertex_data["semantic_pred"] = semantic_pred
        if instance_pred is not None:
            vertex_data["instance_pred"] = instance_pred
        if score is not None:
            vertex_data["score"] = score

        # 6) Write PLY file (world coordinates, matched subset only)
        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el], text=False).write(final_output_path)

        logger.info(f"  Saved: {os.path.basename(final_output_path)}")
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
    
    # Convert LAS/LAZ to PLY if needed for processing
    os.makedirs(CONFIG.TEST_DATA_DIR, exist_ok=True)
    
    if file_ext in ['.las', '.laz']:
        ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
        logger.info(f"  Converting to PLY...")
        if not convert_las_to_ply(input_file, ply_file):
            logger.error(f"Failed to convert {input_file}")
            return False
    elif file_ext == '.ply':
        ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
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

def run_inference_on_files(input_files, model_path=None, config_file=None, output_dir=None, cuda_device="0", keep_intermediate=False):
    """
    Run inference on multiple files programmatically.
    
    Args:
        input_files: List of input file paths (PLY or LAS/LAZ)
        model_path: Path to model checkpoint (None = auto-detect)
        config_file: Path to config file (None = use default)
        output_dir: Output directory (None = create timestamped directory)
        cuda_device: CUDA device ID
        keep_intermediate: Whether to keep intermediate files
        
    Returns:
        str: Path to output directory containing predictions
    """
    # Change to work directory
    os.chdir(CONFIG.WORK_DIR)
    
    # Find model checkpoint
    model_path = find_model_checkpoint(model_path)
    
    # Get config file
    if config_file is None:
        config_file = CONFIG.CONFIG_FILE
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
  # Process single file
  python infer_forestformer3d.py --input file.ply
  
  # Process multiple files
  python infer_forestformer3d.py --input file1.ply file2.las file3.ply
  
  # Process all files in a directory
  python infer_forestformer3d.py --input directory/
  
  # Custom output directory
  python infer_forestformer3d.py --input file.ply --output custom_output/
  
  # Custom model checkpoint
  python infer_forestformer3d.py --input file.ply --model path/to/model.pth
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
        help="Path to model checkpoint .pth file (optional)"
    )
    parser.add_argument(
        "--config",
        help=f"Path to config file (default: {CONFIG.CONFIG_FILE})"
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device ID (default: 0)"
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
            cuda_device=args.cuda_device,
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

