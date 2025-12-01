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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        logger.info(f"✅ Converted to: {output_file}")
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
    logger.info("[Step 0] Cleaning temporary data files...")
    
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
    logger.info("[Step 1] Generating test_list.txt...")
    
    os.makedirs(CONFIG.META_DIR, exist_ok=True)
    test_list = os.path.join(CONFIG.META_DIR, "test_list.txt")
    test_list_init = os.path.join(CONFIG.META_DIR, "test_list_initial.txt")
    
    with open(test_list, 'w') as f:
        f.write(f"{base_name}\n")
    
    shutil.copy(test_list, test_list_init)
    logger.info(f"✅ Created test_list.txt with: {base_name}")

def run_preprocessing():
    """Run data preprocessing steps."""
    logger.info("[Step 2] Running data preprocessing...")
    
    # Change to data directory and run batch_load
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
        logger.info("✅ Data loading complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data loading failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def create_data_info():
    """Create data info files."""
    logger.info("[Step 3] Creating data info files...")
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(CONFIG.WORK_DIR, "tools/create_data_forainetv2.py"), "forainetv2"],
            cwd=CONFIG.WORK_DIR,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Data info files created")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data info creation failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def run_inference(config_file, model_path, cuda_device="0"):
    """Run ForestFormer3D inference."""
    logger.info("[Step 4] Running inference...")
    
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
        logger.info("✅ Inference complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def copy_final_predictions(base_name, intermediate_output_dir, final_output_dir):
    """
    Copy final prediction file from intermediate output to final output directory.
    Skips merge step for semantic segmentation - uses raw _1.ply file directly.
    Applies coordinate offsets to restore original coordinate system.
    """
    # Look for the raw output file
    pred_file = os.path.join(intermediate_output_dir, f"{base_name}_1.ply")
    
    if not os.path.exists(pred_file):
        logger.warning(f"Output file not found at {pred_file}")
        logger.info("Checking for alternative output locations...")
        alt_files = glob.glob(os.path.join(intermediate_output_dir, f"*{base_name}*"))
        if alt_files:
            logger.info(f"Found: {alt_files[:5]}")
        return False
    
    # Load offsets to restore original coordinates
    offset_file = os.path.join(CONFIG.WORK_DIR, 'data/ForAINetV2/forainetv2_instance_data', f'{base_name}_offsets.npy')
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
        # Load PLY file
        from plyfile import PlyData, PlyElement
        
        ply_data = PlyData.read(pred_file)
        vertex = ply_data['vertex'].data
        
        # Extract coordinates
        points = np.vstack((vertex['x'], vertex['y'], vertex['z'])).transpose()
        
        # Apply offsets to restore original coordinate system
        if offsets is not None:
            logger.info("Applying coordinate offsets to restore original coordinate system...")
            points[:, 0] += offsets[0]
            points[:, 1] += offsets[1]
            points[:, 2] += offsets[2]
        
        # Create new vertex data with corrected coordinates
        dtype_list = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        field_data = {
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2]
        }
        
        # Add all other fields from original file
        for field_name in vertex.dtype.names:
            if field_name not in ['x', 'y', 'z']:
                dtype_list.append((field_name, vertex.dtype[field_name]))
                field_data[field_name] = vertex[field_name]
        
        # Create structured array
        vertex_data = np.empty(len(points), dtype=dtype_list)
        for field_name, field_values in field_data.items():
            vertex_data[field_name] = field_values
        
        # Write PLY file with corrected coordinates
        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el], text=False).write(final_output_path)
        
        logger.info(f"✅ Saved final prediction with corrected coordinates to: {final_output_path}")
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {os.path.basename(input_file)}")
    logger.info(f"{'='*60}")
    
    # Convert LAS/LAZ to PLY if needed
    os.makedirs(CONFIG.TEST_DATA_DIR, exist_ok=True)
    
    if file_ext in ['.las', '.laz']:
        ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
        if not convert_las_to_ply(input_file, ply_file):
            logger.error(f"Failed to convert {input_file}")
            return False
    elif file_ext == '.ply':
        ply_file = os.path.join(CONFIG.TEST_DATA_DIR, f"{base_name}.ply")
        logger.info(f"Copying PLY file to test_data directory...")
        shutil.copy2(input_file, ply_file)
        logger.info(f"✅ Copied to: {ply_file}")
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
    if not run_inference(config_file, model_path, cuda_device):
        return False
    
    # Copy final prediction (skip merge step for semantic segmentation)
    if not copy_final_predictions(base_name, intermediate_output_dir, final_output_dir):
        logger.warning(f"Failed to copy final prediction for {base_name}")
        return False
    
    # Clean up intermediate files if requested
    if not keep_intermediate:
        logger.info("Cleaning up intermediate files...")
        cleanup_intermediate_files(base_name, intermediate_output_dir)
    
    # Print results
    final_output_file = os.path.join(final_output_dir, f"{base_name}_classified_by_ff3d.ply")
    logger.info(f"\n✅ Inference complete for {base_name}!")
    logger.info(f"Final output: {final_output_file}")
    
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
    logger.info(f"Using model checkpoint: {model_path}")
    
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
    logger.info(f"Final output: {output_dir}")
    
    # Process each file
    for i, input_file in enumerate(input_files, 1):
        logger.info(f"\n[{i}/{len(input_files)}] Processing file...")
        
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
    
    logger.info(f"\n✅ Inference complete for {len(input_files)} file(s)")
    logger.info(f"Results saved to: {output_dir}")
    
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete!")
        logger.info(f"  Final results: {output_dir}")
        logger.info(f"{'='*60}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

