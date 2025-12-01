"""
Data loading utilities for point cloud files with ground truth labels.
"""

import os
import numpy as np
import logging
import laspy
from plyfile import PlyData

logger = logging.getLogger(__name__)

def find_field_with_fallback(available_fields, primary_names, fallback_names=None):
    """
    Find a field name from available fields, supporting both primary names and fallback names.
    
    Args:
        available_fields: List of field names available in the PLY file
        primary_names: List of preferred field names to look for first
        fallback_names: List of fallback field names to try if primary names not found
    
    Returns:
        str: The found field name, or None if no match found
    """
    for name in primary_names:
        if name in available_fields:
            return name
    
    if fallback_names:
        for name in fallback_names:
            if name in available_fields:
                return name
    
    return None

def load_labeled_ply(file_path, downsample_ratio=1.0, return_scanner_features=False, verbose=False):
    """
    Load labeled point cloud data from PLY format.
    
    Args:
        file_path: Path to the labeled PLY file
        downsample_ratio: Fraction of points to keep (1.0 = all points)
        return_scanner_features: Whether to return scanner features
        verbose: Whether to show detailed progress logs
        
    Returns:
        If return_scanner_features is False:
            - Nx3 numpy array of XYZ points
            - N numpy array of labels (0=branch, 1=leaf)
        If return_scanner_features is True:
            - Nx3 numpy array of XYZ points
            - N numpy array of labels (0=branch, 1=leaf)
            - N numpy array of intensity values (or None)
            - N numpy array of return numbers (or None)
            - N numpy array of number of returns (or None)
    """
    file_name = os.path.basename(file_path)
    
    try:
        ply_data = PlyData.read(file_path)
        vertex = ply_data['vertex'].data
        
        # Extract coordinates
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.vstack((x, y, z)).transpose()
        
        # Get classification values
        available_fields = vertex.dtype.names
        classification_field = find_field_with_fallback(
            available_fields, 
            primary_names=['classification'], 
            fallback_names=['scalar_Classification']
        )
        
        if classification_field:
            labels = vertex[classification_field].astype(int)
        else:
            raise ValueError(f"No 'classification' or 'scalar_Classification' field found in PLY file. Available fields: {', '.join(available_fields)}")
        
        # Get scanner features if requested
        intensity_field = find_field_with_fallback(available_fields, ['intensity'], ['scalar_Intensity'])
        return_number_field = find_field_with_fallback(available_fields, ['return_number'], ['scalar_Return_Number'])
        number_of_returns_field = find_field_with_fallback(available_fields, ['number_of_returns'], ['scalar_Number_Of_Returns'])
        
        scanner_features_available = (intensity_field is not None and 
                                     return_number_field is not None and 
                                     number_of_returns_field is not None)
        
        if return_scanner_features:
            intensity = vertex[intensity_field] if intensity_field else None
            return_number = vertex[return_number_field] if return_number_field else None
            number_of_returns = vertex[number_of_returns_field] if number_of_returns_field else None
            
            if not scanner_features_available and verbose:
                logger.warning(f"Scanner features requested but not all available in file {file_name}")
        
        # Apply downsampling if needed
        if downsample_ratio < 1.0:
            num_points = len(points)
            sample_size = int(num_points * downsample_ratio)
            indices = np.random.choice(num_points, sample_size, replace=False)
            points = points[indices]
            labels = labels[indices]
            if return_scanner_features:
                if intensity is not None:
                    intensity = intensity[indices]
                if return_number is not None:
                    return_number = return_number[indices]
                if number_of_returns is not None:
                    number_of_returns = number_of_returns[indices]
        
        if return_scanner_features:
            return points, labels, intensity, return_number, number_of_returns
        else:
            return points, labels
    except Exception as e:
        logger.error(f"Error loading PLY file {file_name}: {str(e)}")
        raise

def load_labeled_las(file_path, downsample_ratio=1.0, return_scanner_features=False, verbose=False):
    """
    Load labeled point cloud data from LAS/LAZ format.
    
    Args:
        file_path: Path to the labeled LAS/LAZ file
        downsample_ratio: Fraction of points to keep (1.0 = all points)
        return_scanner_features: Whether to return scanner features
        verbose: Whether to show detailed progress logs
        
    Returns:
        If return_scanner_features is False:
            - Nx3 numpy array of XYZ points
            - N numpy array of labels (0=branch, 1=leaf)
        If return_scanner_features is True:
            - Nx3 numpy array of XYZ points
            - N numpy array of labels (0=branch, 1=leaf)
            - N numpy array of intensity values (or None)
            - N numpy array of return numbers (or None)
            - N numpy array of number of returns (or None)
    """
    file_name = os.path.basename(file_path)
    
    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Extract classification values
        if hasattr(las, 'classification'):
            original_labels = np.array(las.classification, dtype=int)
        else:
            raise ValueError(f"No 'classification' field found in LAS file {file_name}")
        
        # Check if the file already uses 0/1 encoding (direct branch/leaf labels)
        unique_original = np.unique(original_labels)
        if set(unique_original).issubset({0, 1}):
            labels = original_labels.copy()
        else:
            # Map LAS classification codes to binary branch/leaf labels
            # Vegetation codes (3,4,5) = leaves, everything else = branches
            leaf_codes = {3, 4, 5}
            labels = np.array([1 if code in leaf_codes else 0 for code in original_labels])
        
        # Get scanner features
        scanner_features_available = (hasattr(las, 'intensity') and 
                                     hasattr(las, 'return_number') and 
                                     hasattr(las, 'number_of_returns'))
        
        if return_scanner_features:
            intensity = np.array(las.intensity) if hasattr(las, 'intensity') else None
            return_number = np.array(las.return_number) if hasattr(las, 'return_number') else None
            number_of_returns = np.array(las.number_of_returns) if hasattr(las, 'number_of_returns') else None
            
            if not scanner_features_available and verbose:
                logger.warning(f"Scanner features requested but not all available in file {file_name}")
        
        # Apply downsampling if needed
        if downsample_ratio < 1.0:
            num_points = len(points)
            sample_size = int(num_points * downsample_ratio)
            indices = np.random.choice(num_points, sample_size, replace=False)
            points = points[indices]
            labels = labels[indices]
            if return_scanner_features:
                if intensity is not None:
                    intensity = intensity[indices]
                if return_number is not None:
                    return_number = return_number[indices]
                if number_of_returns is not None:
                    number_of_returns = number_of_returns[indices]
        
        if return_scanner_features:
            return points, labels, intensity, return_number, number_of_returns
        else:
            return points, labels
    except Exception as e:
        logger.error(f"Error loading LAS file {file_name}: {str(e)}")
        raise

