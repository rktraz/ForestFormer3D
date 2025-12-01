"""
Evaluation metrics for semantic segmentation (branches/leaves).
"""

import numpy as np
import logging
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

logger = logging.getLogger(__name__)

def calculate_height_stratified_metrics(points, true_labels, predictions):
    """
    Calculate metrics stratified by height percentiles.
    
    Args:
        points: 3D point cloud array (N x 3) with X, Y, Z coordinates
        true_labels: Ground truth labels array (N,)
        predictions: Predicted labels array (N,)
        
    Returns:
        dict: Metrics for lower, middle, and upper thirds of the tree
    """
    # Get Z coordinates (height)
    heights = points[:, 2]
    total_points = len(points)
    
    # Calculate height percentiles
    min_height = np.min(heights)
    max_height = np.max(heights)
    height_range = max_height - min_height
    
    # Define strata boundaries (0-33%, 33-66%, 66-100%)
    lower_bound = min_height + height_range * 0.33
    middle_bound = min_height + height_range * 0.66
    
    # Create masks for each stratum
    lower_mask = heights < lower_bound
    middle_mask = (heights >= lower_bound) & (heights < middle_bound)
    upper_mask = heights >= middle_bound
    
    # Calculate metrics for each stratum
    strata_metrics = {}
    
    for name, mask in [('lower', lower_mask), ('middle', middle_mask), ('upper', upper_mask)]:
        if np.sum(mask) == 0:
            strata_metrics[name] = {
                'f1': 0, 'branch_f1': 0, 'leaf_f1': 0,
                'branch_precision': 0, 'branch_recall': 0, 'branch_jaccard': 0,
                'leaf_precision': 0, 'leaf_recall': 0, 'leaf_jaccard': 0,
                'jaccard_index': 0, 'accuracy': 0, 'point_percent': 0
            }
            continue
        
        # Calculate comprehensive metrics for this stratum
        stratum_true = true_labels[mask]
        stratum_pred = predictions[mask]
        
        # Calculate per-class metrics
        branch_f1_stratum = f1_score(stratum_true, stratum_pred, pos_label=0, zero_division=0)
        leaf_f1_stratum = f1_score(stratum_true, stratum_pred, pos_label=1, zero_division=0)
        branch_jaccard_stratum = jaccard_score(stratum_true, stratum_pred, pos_label=0, zero_division=0)
        leaf_jaccard_stratum = jaccard_score(stratum_true, stratum_pred, pos_label=1, zero_division=0)
        
        strata_metrics[name] = {
            'f1': f1_score(stratum_true, stratum_pred, average='macro', zero_division=0),
            'branch_f1': branch_f1_stratum,
            'leaf_f1': leaf_f1_stratum,
            'branch_precision': precision_score(stratum_true, stratum_pred, pos_label=0, zero_division=0),
            'branch_recall': recall_score(stratum_true, stratum_pred, pos_label=0, zero_division=0),
            'branch_jaccard': branch_jaccard_stratum,
            'leaf_precision': precision_score(stratum_true, stratum_pred, pos_label=1, zero_division=0),
            'leaf_recall': recall_score(stratum_true, stratum_pred, pos_label=1, zero_division=0),
            'leaf_jaccard': leaf_jaccard_stratum,
            'jaccard_index': jaccard_score(stratum_true, stratum_pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(stratum_true, stratum_pred),
            'point_percent': np.sum(mask) / total_points * 100
        }
    
    return strata_metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive segmentation metrics.
    
    Args:
        y_true: Ground truth labels array (N,)
        y_pred: Predicted labels array (N,)
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    if len(y_true) == 0:
        logger.error("Empty ground truth array provided to metrics computation")
        return {}
    
    # Calculate per-class metrics
    branch_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    branch_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    branch_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    branch_jaccard = jaccard_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    leaf_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    leaf_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    leaf_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    leaf_jaccard = jaccard_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        # Overall metrics: macro-averaged
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'jaccard_index': jaccard_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Add per-class metrics
    metrics['branch_precision'] = branch_precision
    metrics['branch_recall'] = branch_recall
    metrics['branch_f1'] = branch_f1
    metrics['branch_jaccard'] = branch_jaccard
    metrics['leaf_precision'] = leaf_precision
    metrics['leaf_recall'] = leaf_recall
    metrics['leaf_f1'] = leaf_f1
    metrics['leaf_jaccard'] = leaf_jaccard
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'branch_predicted_branch': int(cm[0, 0]),
            'branch_predicted_leaf': int(cm[0, 1]),
            'leaf_predicted_branch': int(cm[1, 0]),
            'leaf_predicted_leaf': int(cm[1, 1]),
        }
    
    # Detailed classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, 
                                                          target_names=['branch', 'leaf'],
                                                          output_dict=True)
    
    return metrics

