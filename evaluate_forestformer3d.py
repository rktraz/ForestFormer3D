"""
ForestFormer3D Evaluation Module

This script evaluates ForestFormer3D model predictions on test files with ground truth labels.
It loads predictions from ForestFormer3D output PLY files, matches them with ground truth,
calculates comprehensive metrics, and logs results to Weights & Biases.

Usage:
    python evaluate_forestformer3d.py [--output <output_dir>] [--save-classified-clouds] [--verbose]
    
Arguments:
    --output              : Output directory for evaluation results (default: evaluation_results)
    --save-classified-clouds : Save classified point cloud files (default: False)
    --verbose             : Show detailed progress logs
"""

import os
import sys
import glob
import numpy as np
import logging
import argparse
import traceback
import time
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree
import wandb
from plyfile import PlyData
import laspy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import data loading and metrics functions from tools
tools_dir = os.path.join(os.path.dirname(__file__), 'tools')
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)

try:
    from data_loading import (
        load_labeled_ply,
        load_labeled_las,
        find_field_with_fallback
    )
    from metrics import (
        calculate_metrics,
        calculate_height_stratified_metrics
    )
except ImportError as e:
    logger.error(f"Could not import required modules from tools/: {e}")
    logger.error("Make sure tools/data_loading.py and tools/metrics.py exist.")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    jaccard_score
)

#####################################################################
# CONFIGURATION
#####################################################################

class CONFIG:
    """Configuration constants for ForestFormer3D evaluation."""
    # Working directory
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Classification values
    CLASS_BRANCH = 0  # Classification value for branches
    CLASS_LEAF = 1    # Classification value for leaves
    
    # ForestFormer3D semantic label mapping
    # 0,1 = branches -> map to 0
    # 2 = leaves -> map to 1
    FF3D_BRANCH_VALUES = [0, 1]
    FF3D_LEAF_VALUE = 2
    
    # Base directory for evaluation outputs
    EVALUATION_OUTPUT_DIR = "evaluation_results"
    
    # Default spatial matching tolerance (meters)
    SPATIAL_MATCH_TOLERANCE = 0.01  # 1cm tolerance for matching points
    
    # Default model checkpoint path (can be overridden)
    DEFAULT_MODEL_CHECKPOINT = "work_dirs/clean_forestformer/epoch_3000_fix.pth"
    
    # Prediction output directories
    # Can be set to inference_runs/run_TIMESTAMP/ or work_dirs/output/round_1
    PREDICTION_OUTPUT_DIR = "inference_runs"  # Will search for *_classified_by_ff3d.ply files
    USE_TIMESTAMPED_RUNS = True  # Look for files in inference_runs/run_*/ directories
    
    # List of evaluation files (ground truth paths)
    EVALUATION_FILES = [
        "evaluation_files/107_labeled.las",
        "evaluation_files/129_labeled.ply",
        "evaluation_files/236_labeled.las",
        "evaluation_files/245_labeled.las",
        "evaluation_files/251_labeled.las",
        "evaluation_files/FagSyl_BR05_P8T4_2019-07-10_q2_TLS-on_c.ply",
        "evaluation_files/PinSyl_KA10_03_2019-07-30_q2_TLS-on_c.ply",
        "evaluation_files/QueRub_KA09_T053_2019-08-20_q1_TLS-on_c.ply"
    ]

#####################################################################
# DATA LOADING FUNCTIONS
#####################################################################

def load_forestformer3d_prediction(ply_path):
    """
    Load ForestFormer3D prediction from PLY file.
    
    Args:
        ply_path: Path to PLY file with ForestFormer3D predictions
        
    Returns:
        tuple: (points, semantic_pred) where:
            - points: Nx3 numpy array of XYZ coordinates
            - semantic_pred: N numpy array of semantic predictions (0,1=branch, 2=leaf)
    """
    try:
        ply_data = PlyData.read(ply_path)
        vertex = ply_data['vertex'].data
        
        # Extract coordinates
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.vstack((x, y, z)).transpose()
        
        # Extract semantic predictions
        if 'semantic_pred' not in vertex.dtype.names:
            raise ValueError(f"No 'semantic_pred' field found in PLY file: {ply_path}")
        
        semantic_pred = vertex['semantic_pred'].astype(int)
        
        logger.info(f"Loaded {len(points)} points from ForestFormer3D prediction: {os.path.basename(ply_path)}")
        logger.info(f"Semantic prediction distribution: {dict(zip(*np.unique(semantic_pred, return_counts=True)))}")
        
        return points, semantic_pred
        
    except Exception as e:
        logger.error(f"Error loading ForestFormer3D prediction from {ply_path}: {str(e)}")
        raise

def map_forestformer3d_labels(semantic_pred):
    """
    Map ForestFormer3D semantic labels to standard format.
    
    Args:
        semantic_pred: Numpy array of ForestFormer3D semantic predictions (0,1,2)
        
    Returns:
        Numpy array with mapped labels (0=branch, 1=leaf)
    """
    # Map: 0,1 -> 0 (branch), 2 -> 1 (leaf)
    mapped = np.zeros_like(semantic_pred, dtype=int)
    mapped[semantic_pred == CONFIG.FF3D_LEAF_VALUE] = CONFIG.CLASS_LEAF
    # Values 0 and 1 are already 0 (branch) in the mapped array
    
    return mapped

def match_points_by_order(pred_points, true_points, true_labels):
    """
    Match points by index order (1:1 matching) when point counts match exactly.
    Assumes same input file = same point order.
    
    Args:
        pred_points: Nx3 array of predicted point coordinates
        true_points: Nx3 array of ground truth point coordinates
        true_labels: N array of ground truth labels
        
    Returns:
        tuple: (matched_points, matched_labels)
    """
    return pred_points, true_labels.copy()

def spatial_match_points(pred_points, true_points, true_labels, tolerance=CONFIG.SPATIAL_MATCH_TOLERANCE):
    """
    Match predicted points to ground truth points using spatial nearest neighbor matching.
    
    Args:
        pred_points: Nx3 array of predicted point coordinates
        true_points: Mx3 array of ground truth point coordinates
        true_labels: M array of ground truth labels
        tolerance: Maximum distance for matching (in meters)
        
    Returns:
        tuple: (matched_points, matched_labels) where:
            - matched_points: Nx3 array of matched point coordinates
            - matched_labels: N array of matched ground truth labels
    """
    logger.info(f"Matching {len(pred_points)} prediction points to {len(true_points)} ground truth points...")
    
    # Build KDTree from ground truth points
    tree = cKDTree(true_points)
    
    # Find nearest neighbors for each prediction point
    distances, indices = tree.query(pred_points, k=1)
    
    # Filter out matches that are too far away
    valid_mask = distances <= tolerance
    num_valid = np.sum(valid_mask)
    num_invalid = len(pred_points) - num_valid
    
    if num_invalid > 0:
        logger.warning(f"{num_invalid} prediction points could not be matched within {tolerance}m tolerance")
        logger.warning(f"Max distance: {np.max(distances):.4f}m, Mean distance: {np.mean(distances):.4f}m")
    
    # Get matched labels
    matched_labels = true_labels[indices]
    matched_labels[~valid_mask] = -1  # Mark unmatched points with -1
    
    logger.info(f"Successfully matched {num_valid}/{len(pred_points)} points")
    
    return pred_points, matched_labels

#####################################################################
# EVALUATION FUNCTIONS
#####################################################################

def evaluate_file(pred_ply_path, gt_file_path, output_dir=None, save_classified_cloud=False, verbose=False):
    """
    Evaluate a single file by comparing ForestFormer3D predictions with ground truth.
    
    Args:
        pred_ply_path: Path to ForestFormer3D prediction PLY file
        gt_file_path: Path to ground truth file (PLY or LAS)
        output_dir: Output directory for saving classified clouds (optional)
        save_classified_cloud: Whether to save classified point cloud file
        verbose: Whether to show detailed progress logs
        
    Returns:
        dict: Evaluation results with metrics and statistics
    """
    try:
        logger.info(f"Evaluating: {os.path.basename(pred_ply_path)} vs {os.path.basename(gt_file_path)}")
        
        # Load ForestFormer3D predictions
        pred_points, semantic_pred_ff3d = load_forestformer3d_prediction(pred_ply_path)
        
        # Map ForestFormer3D labels to standard format (0=branch, 1=leaf)
        predictions = map_forestformer3d_labels(semantic_pred_ff3d)
        
        # Load ground truth
        file_ext = os.path.splitext(gt_file_path)[1].lower()
        if file_ext == '.ply':
            true_points, true_labels = load_labeled_ply(gt_file_path, verbose=verbose)
        elif file_ext in ['.las', '.laz']:
            true_points, true_labels = load_labeled_las(gt_file_path, verbose=verbose)
        else:
            logger.error(f"Unsupported ground truth file format: {file_ext}")
            return None
        
        logger.info(f"Loaded {len(true_points)} ground truth points")
        logger.info(f"Ground truth distribution: {dict(zip(*np.unique(true_labels, return_counts=True)))}")
        
        # Match points
        # If point counts match exactly, use 1:1 order-based matching (same input file = same order)
        # Otherwise, use spatial matching
        if len(pred_points) == len(true_points):
            logger.info(f"Point counts match exactly ({len(pred_points)}). Using 1:1 order-based matching.")
            matched_points, matched_labels = match_points_by_order(pred_points, true_points, true_labels)
            logger.info(f"Matched {len(matched_labels)} points by order (1:1 correspondence)")
        else:
            logger.info(f"Point counts differ (pred: {len(pred_points)}, gt: {len(true_points)}). Using spatial matching.")
            matched_points, matched_labels = spatial_match_points(pred_points, true_points, true_labels)
            
            # Filter out unmatched points (labels == -1)
            valid_mask = matched_labels != -1
            if not np.all(valid_mask):
                num_filtered = np.sum(~valid_mask)
                logger.warning(f"Filtering out {num_filtered} unmatched points from evaluation")
                matched_points = matched_points[valid_mask]
                predictions = predictions[valid_mask]
                matched_labels = matched_labels[valid_mask]
        
        if len(matched_labels) == 0:
            logger.error("No valid matched points for evaluation. Skipping this file.")
            return None
        
        # Final verification
        if len(matched_labels) != len(predictions):
            logger.error(f"Length mismatch after filtering: labels={len(matched_labels)}, predictions={len(predictions)}")
            return None
        
        # Calculate metrics
        metrics = calculate_metrics(matched_labels, predictions)
        
        # Calculate height-stratified metrics
        height_metrics = calculate_height_stratified_metrics(matched_points, matched_labels, predictions)
        
        # Log summary
        logger.info(f"Evaluation complete: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # Save classified cloud if requested
        output_file = None
        if save_classified_cloud and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(gt_file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_classified.ply")
            
            # Save as PLY with predictions
            from plyfile import PlyElement
            dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('classification', 'i4')]
            vertex = np.array([tuple(matched_points[i]) + (predictions[i],) for i in range(len(matched_points))], dtype=dtype)
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el], text=False).write(output_file)
            logger.info(f"Saved classified cloud to: {output_file}")
        
        return {
            'pred_file': pred_ply_path,
            'gt_file': gt_file_path,
            'file_name': os.path.basename(gt_file_path),
            'total_points': len(matched_points),
            'true_branch_points': np.sum(matched_labels == CONFIG.CLASS_BRANCH),
            'true_leaf_points': np.sum(matched_labels == CONFIG.CLASS_LEAF),
            'pred_branch_points': np.sum(predictions == CONFIG.CLASS_BRANCH),
            'pred_leaf_points': np.sum(predictions == CONFIG.CLASS_LEAF),
            'metrics': metrics,
            'height_metrics': height_metrics,
            'true_labels': matched_labels,
            'predictions': predictions,
            'points': matched_points,
            'output_file': output_file
        }
        
    except Exception as e:
        logger.error(f"Error evaluating file {pred_ply_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

#####################################################################
# W&B LOGGING
#####################################################################

def log_evaluation_to_wandb(model_checkpoint_path, results, runtime_seconds, output_dir):
    """Log evaluation results to Weights & Biases."""
    try:
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            logger.error("No valid results to log to W&B")
            return None
        
        # Collect evaluation file names
        evaluation_files = [r['file_name'] for r in valid_results]
        
        # Get aggregate metrics using numpy arrays
        all_true_labels = np.concatenate([r['true_labels'] for r in valid_results])
        all_pred_labels = np.concatenate([r['predictions'] for r in valid_results])
        
        # Calculate aggregate metrics
        agg_metrics = calculate_metrics(all_true_labels, all_pred_labels)
        
        # Get height metrics from first result
        height_metrics = valid_results[0]['height_metrics']
        
        # Prepare W&B config
        wandb_config = {
            # Model metadata
            "model_type": "ForestFormer3D",
            "model_checkpoint_path": model_checkpoint_path,
            
            # Evaluation information
            "evaluation_files": evaluation_files,
            "total_test_points": sum(r['total_points'] for r in valid_results),
            "evaluation_runtime": runtime_seconds,
        }
        
        # Initialize W&B
        wandb.init(
            project="tree-segmentation",
            config=wandb_config,
            settings=wandb.Settings(console="off")
        )
        
        # Metrics dictionary
        metrics_dict = {
            # Standard metrics
            "accuracy": agg_metrics["accuracy"],
            "precision": agg_metrics["precision"],
            "recall": agg_metrics["recall"],
            "f1": agg_metrics["f1"],
            "jaccard_index": agg_metrics["jaccard_index"],
            "branch_precision": agg_metrics["branch_precision"],
            "branch_recall": agg_metrics["branch_recall"],
            "branch_f1": agg_metrics["branch_f1"],
            "branch_jaccard": agg_metrics["branch_jaccard"],
            "leaf_precision": agg_metrics["leaf_precision"],
            "leaf_recall": agg_metrics["leaf_recall"],
            "leaf_f1": agg_metrics["leaf_f1"],
            "leaf_jaccard": agg_metrics["leaf_jaccard"],
            
            # Height-stratified metrics
            "lower_f1": height_metrics["lower"]["f1"],
            "lower_branch_f1": height_metrics["lower"]["branch_f1"],
            "lower_stratum_share": height_metrics["lower"]["point_percent"] / 100,
            "middle_f1": height_metrics["middle"]["f1"],
            "middle_branch_f1": height_metrics["middle"]["branch_f1"],
            "middle_stratum_share": height_metrics["middle"]["point_percent"] / 100,
            "upper_f1": height_metrics["upper"]["f1"],
            "upper_branch_f1": height_metrics["upper"]["branch_f1"],
            "upper_stratum_share": height_metrics["upper"]["point_percent"] / 100,
            
            # Confusion matrix
            "branch_correctly_classified": agg_metrics["confusion_matrix"]["branch_predicted_branch"],
            "branch_misclassified": agg_metrics["confusion_matrix"]["branch_predicted_leaf"],
            "leaf_correctly_classified": agg_metrics["confusion_matrix"]["leaf_predicted_leaf"],
            "leaf_misclassified": agg_metrics["confusion_matrix"]["leaf_predicted_branch"],
        }
        
        # Log metrics
        wandb.log(metrics_dict)
        
        run_name = wandb.run.name if wandb.run else None
        logger.info(f"Successfully logged evaluation results to W&B (run: {run_name})")
        
        return run_name
        
    except Exception as e:
        logger.error(f"Error logging evaluation to W&B: {str(e)}")
        logger.error(traceback.format_exc())
        return None

#####################################################################
# REPORT GENERATION
#####################################################################

def generate_evaluation_report(results, model_checkpoint, output_dir, runtime_seconds):
    """Generate a detailed evaluation report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("ForestFormer3D Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runtime: {runtime_seconds:.1f} seconds\n")
        f.write(f"Model checkpoint: {model_checkpoint}\n\n")
        
        valid_results = [r for r in results if r is not None]
        
        # Aggregate metrics
        all_true_labels = np.concatenate([r['true_labels'] for r in valid_results])
        all_pred_labels = np.concatenate([r['predictions'] for r in valid_results])
        agg_metrics = calculate_metrics(all_true_labels, all_pred_labels)
        
        f.write("Aggregate Results Across All Test Files\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total test files: {len(valid_results)}\n")
        f.write(f"Total points evaluated: {sum(r['total_points'] for r in valid_results)}\n\n")
        
        f.write("Aggregate Metrics:\n")
        for key, value in agg_metrics.items():
            if key not in ['confusion_matrix', 'classification_report']:
                f.write(f"- {key.replace('_', ' ').capitalize()}: {value:.4f}\n")
        
        # Per-file results
        f.write("\n" + "=" * 60 + "\n")
        f.write("Results by File\n")
        f.write("=" * 60 + "\n\n")
        
        for result in valid_results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"- Total points: {result['total_points']}\n")
            f.write(f"- F1 Score: {result['metrics']['f1']:.4f}\n")
            f.write(f"- Accuracy: {result['metrics']['accuracy']:.4f}\n")
            f.write("\n")
    
    logger.info(f"Evaluation report saved to: {report_path}")
    return report_path

#####################################################################
# MAIN FUNCTION
#####################################################################

def find_model_checkpoint(default_path=None):
    """
    Find the model checkpoint path.
    Tries default path, then looks for common checkpoint locations.
    """
    if default_path and os.path.exists(default_path):
        return default_path
    
    # Try default path from CONFIG
    if os.path.exists(CONFIG.DEFAULT_MODEL_CHECKPOINT):
        return CONFIG.DEFAULT_MODEL_CHECKPOINT
    
    # Try to find any .pth file in work_dirs/clean_forestformer/
    work_dir = "work_dirs/clean_forestformer"
    if os.path.exists(work_dir):
        pth_files = glob.glob(os.path.join(work_dir, "*.pth"))
        if pth_files:
            # Prefer epoch_3000_fix.pth if exists, otherwise use first found
            preferred = [f for f in pth_files if 'epoch_3000_fix' in f]
            if preferred:
                return preferred[0]
            return pth_files[0]
    
    logger.warning("Could not find model checkpoint. Using default path string.")
    return CONFIG.DEFAULT_MODEL_CHECKPOINT

def main():
    parser = argparse.ArgumentParser(description="Evaluate ForestFormer3D model on test files.")
    parser.add_argument("--output", help="Output directory (default: evaluation_results)")
    parser.add_argument("--save-classified-clouds", action="store_true", 
                       help="Save classified point cloud files (default: False)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress logs")
    parser.add_argument("--model-checkpoint", help="Path to model checkpoint .pth file (optional)")
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device ID for inference (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find model checkpoint
    model_checkpoint = find_model_checkpoint(args.model_checkpoint)
    logger.info(f"Using model checkpoint: {model_checkpoint}")
    
    # Get prediction output directory
    if CONFIG.USE_TIMESTAMPED_RUNS:
        # Look for files in inference_runs/run_*/ directories
        pred_output_dir = os.path.join(CONFIG.WORK_DIR, CONFIG.PREDICTION_OUTPUT_DIR)
        logger.info(f"Searching for predictions in: {pred_output_dir}")
    else:
        pred_output_dir = os.path.join(CONFIG.WORK_DIR, CONFIG.PREDICTION_OUTPUT_DIR)
        logger.info(f"Using prediction directory: {pred_output_dir}")
    
    # Get evaluation files from CONFIG
    evaluation_files = CONFIG.EVALUATION_FILES
    if not evaluation_files:
        logger.error("No evaluation files specified in CONFIG.EVALUATION_FILES")
        logger.error("Please add evaluation file paths to CONFIG.EVALUATION_FILES in the script.")
        sys.exit(1)
    
    # Filter out files that don't exist
    valid_evaluation_files = []
    for gt_file_path in evaluation_files:
        if not os.path.exists(gt_file_path):
            logger.warning(f"Ground truth file not found: {gt_file_path}. Skipping.")
            continue
        valid_evaluation_files.append(gt_file_path)
    
    if not valid_evaluation_files:
        logger.error("No valid evaluation files found. Exiting.")
        sys.exit(1)
    
    # Check which files need inference
    files_needing_inference = []
    for gt_file_path in valid_evaluation_files:
        file_name = os.path.basename(gt_file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Check if prediction exists
        pred_file = None
        if CONFIG.USE_TIMESTAMPED_RUNS:
            run_dirs = glob.glob(os.path.join(pred_output_dir, "run_*"))
            for run_dir in sorted(run_dirs, reverse=True):
                pattern = os.path.join(run_dir, f"{base_name}_classified_by_ff3d.ply")
                matching = glob.glob(pattern)
                if matching:
                    pred_file = matching[0]
                    break
        
        if not pred_file:
            files_needing_inference.append(gt_file_path)
    
    # Run inference on files that need it
    if files_needing_inference:
        logger.info(f"Running inference on {len(files_needing_inference)} file(s) that don't have predictions...")
        
        # Import inference function
        infer_script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, infer_script_dir)
        from infer_forestformer3d import run_inference_on_files
        
        try:
            inference_output_dir = run_inference_on_files(
                input_files=files_needing_inference,
                model_path=model_checkpoint,
                config_file=None,  # Use default
                output_dir=None,  # Create timestamped directory
                cuda_device=args.cuda_device,
                keep_intermediate=False
            )
            logger.info(f"Inference complete. Predictions saved to: {inference_output_dir}")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    # Create output directory for evaluation results
    output_dir = args.output if args.output else CONFIG.EVALUATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    results = []
    start_time = time.time()
    
    for gt_file_path in valid_evaluation_files:
        file_name = os.path.basename(gt_file_path)
        
        # Find corresponding prediction file
        base_name = os.path.splitext(file_name)[0]
        
        # Try to find the new naming convention first: *_classified_by_ff3d.ply
        pred_file = None
        
        if CONFIG.USE_TIMESTAMPED_RUNS:
            # Search in inference_runs/run_*/ directories
            run_dirs = glob.glob(os.path.join(pred_output_dir, "run_*"))
            for run_dir in sorted(run_dirs, reverse=True):  # Start with most recent
                pattern = os.path.join(run_dir, f"{base_name}_classified_by_ff3d.ply")
                matching = glob.glob(pattern)
                if matching:
                    pred_file = matching[0]
                    break
        else:
            # Legacy: look in work_dirs/output/round_1
            pattern = os.path.join(pred_output_dir, f"{base_name}_classified_by_ff3d.ply")
            matching = glob.glob(pattern)
            if matching:
                pred_file = matching[0]
        
        # Fallback: try old naming conventions for backward compatibility
        if not pred_file:
            # Try *_round*.ply pattern (legacy)
            if CONFIG.USE_TIMESTAMPED_RUNS:
                for run_dir in sorted(glob.glob(os.path.join(pred_output_dir, "run_*")), reverse=True):
                    pattern = os.path.join(run_dir, f"{base_name}_round*.ply")
                    matching = glob.glob(pattern)
                    if matching:
                        pred_file = sorted(matching)[0]
                        break
            else:
                pattern = os.path.join(pred_output_dir, f"{base_name}_round*.ply")
                matching = glob.glob(pattern)
                if matching:
                    pred_file = sorted(matching)[0]
        
        # Final fallback: try *_1.ply (raw output in work_dirs/output)
        if not pred_file:
            legacy_pattern = os.path.join("work_dirs/output", f"{base_name}_1.ply")
            if os.path.exists(legacy_pattern):
                pred_file = os.path.abspath(legacy_pattern)
        
        if not pred_file:
            logger.error(f"Prediction file not found for {file_name} after inference step")
            sys.exit(1)
        
        logger.info(f"Evaluating: {file_name} -> {os.path.basename(pred_file)}")
        
        result = evaluate_file(
            pred_ply_path=pred_file,
            gt_file_path=gt_file_path,
            output_dir=output_dir if args.save_classified_clouds else None,
            save_classified_cloud=args.save_classified_clouds,
            verbose=args.verbose
        )
        
        if result:
            results.append(result)
    
    runtime_seconds = time.time() - start_time
    
    # Generate report
    if results:
        report_path = generate_evaluation_report(
            results, model_checkpoint, output_dir, runtime_seconds
        )
        
        # Log to W&B
        wandb_run_name = log_evaluation_to_wandb(
            model_checkpoint, results, runtime_seconds, output_dir
        )
        
        if wandb_run_name:
            logger.info(f"W&B run name: {wandb_run_name}")
        
        # Finish W&B run
        wandb.finish()
        
        logger.info(f"Evaluation complete in {runtime_seconds:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
    else:
        logger.error("No successful evaluations")

if __name__ == "__main__":
    main()

