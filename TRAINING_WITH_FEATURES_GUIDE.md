# ForestFormer3D Training Guide: Training from Scratch with Scanner Features

## 📋 Overview

This guide covers training ForestFormer3D **from scratch** with **scanner features** (intensity, return_number, number_of_returns) in addition to coordinates. We train **three models** to compare performance:

1. **Model 1**: Heidelberg dataset only
2. **Model 2**: UWaterloo dataset only
3. **Model 3**: Combined dataset (both Heidelberg + UWaterloo)

**Key differences from coordinate-only training:**
- Input: 6 channels (x, y, z + 3 scanner features) instead of 3
- Semantic classes: 2 (wood, leaf) instead of 3 (no ground)
- Training: From scratch (no pre-trained weights)
- Features: Global normalization applied to scanner features

---

## 📋 Data Requirements

### Input Data Format

Your standardized data must be in **PLY format** with the following structure:

**Required fields:**
- `x`, `y`, `z` (float64): Point coordinates
- `intensity` (uint16): LiDAR intensity (0-65535)
- `return_number` (uint8): Return number (1-6)
- `number_of_returns` (uint8): Number of returns (1-7)
- `semantic_seg` (int32): Semantic labels
  - `1` = wood
  - `2` = leaf
- `treeID` (int32): Instance IDs
  - `0` = background/unannotated
  - `1+` = tree instance IDs

**Data type:** Binary PLY format

**Note:** Your standardized files in `data/standardized/` already have this format! ✅

---

## 📁 Step 1: Prepare Training Data Splits

### 1.1 Run Data Split Script

The script creates train/val splits (80/20) for each scenario:

```bash
# Prepare all three scenarios at once
python tools/prepare_training_splits.py --scenario all

# Or prepare individually:
python tools/prepare_training_splits.py --scenario heidelberg
python tools/prepare_training_splits.py --scenario uwaterloo
python tools/prepare_training_splits.py --scenario combined
```

**What this script does:**
- Creates train/val splits (80/20) for each scenario
- Copies standardized PLY files to `data/ForAINetV2_{scenario}/train_val_data/`
- Creates `meta_data/train_list.txt` and `meta_data/val_list.txt`
- Uses random seed 42 for reproducibility

**Output structure:**
```
data/ForAINetV2_heidelberg/
├── train_val_data/     # PLY files
│   ├── file1.ply
│   └── ...
└── meta_data/
    ├── train_list.txt  # 8 files (80%)
    └── val_list.txt    # 2 files (20%)

data/ForAINetV2_uwaterloo/
├── train_val_data/
└── meta_data/
    ├── train_list.txt  # 14 files (80%)
    └── val_list.txt    # 3 files (20%)

data/ForAINetV2_combined/
├── train_val_data/
└── meta_data/
    ├── train_list.txt  # 22 files (80%)
    └── val_list.txt    # 5 files (20%)
```

### 1.2 Verify Data Splits

Check that files were copied correctly:

```bash
# Check Heidelberg
ls data/ForAINetV2_heidelberg/train_val_data/ | wc -l  # Should be 10
cat data/ForAINetV2_heidelberg/meta_data/train_list.txt
cat data/ForAINetV2_heidelberg/meta_data/val_list.txt

# Check UWaterloo
ls data/ForAINetV2_uwaterloo/train_val_data/ | wc -l  # Should be 17
cat data/ForAINetV2_uwaterloo/meta_data/train_list.txt

# Check Combined
ls data/ForAINetV2_combined/train_val_data/ | wc -l  # Should be 27
cat data/ForAINetV2_combined/meta_data/train_list.txt
```

---

## ⚙️ Step 2: Run Data Preprocessing

For each scenario, run data preprocessing to generate `.npy` and `.pkl` files.

### 2.1 Preprocess Heidelberg Dataset

```bash
cd data/ForAINetV2_heidelberg

# Delete old .npy files if regenerating
rm -rf forainetv2_heidelberg_instance_data/*

# Copy necessary scripts
cp ../ForAINetV2/batch_load_ForAINetV2_data.py .
cp ../ForAINetV2/load_forainetv2_data.py .

# Generate .npy files from PLY files
python batch_load_ForAINetV2_data.py \
    --train_scan_names_file meta_data/train_list.txt \
    --val_scan_names_file meta_data/val_list.txt \
    --test_scan_names_file meta_data/val_list.txt  # Use val as test

# Return to project root
cd ../..
```

### 2.2 Preprocess UWaterloo Dataset

```bash
cd data/ForAINetV2_uwaterloo

# Delete old .npy files if regenerating
rm -rf forainetv2_uwaterloo_instance_data/*

# Copy necessary scripts
cp ../ForAINetV2/batch_load_ForAINetV2_data.py .
cp ../ForAINetV2/load_forainetv2_data.py .

# Generate .npy files
python batch_load_ForAINetV2_data.py \
    --train_scan_names_file meta_data/train_list.txt \
    --val_scan_names_file meta_data/val_list.txt \
    --test_scan_names_file meta_data/val_list.txt

cd ../..
```

### 2.3 Preprocess Combined Dataset

```bash
cd data/ForAINetV2_combined

# Delete old .npy files if regenerating
rm -rf forainetv2_combined_instance_data/*

# Copy necessary scripts
cp ../ForAINetV2/batch_load_ForAINetV2_data.py .
cp ../ForAINetV2/load_forainetv2_data.py .

# Generate .npy files
python batch_load_ForAINetV2_data.py \
    --train_scan_names_file meta_data/train_list.txt \
    --val_scan_names_file meta_data/val_list.txt \
    --test_scan_names_file meta_data/val_list.txt

cd ../..
```

### 2.4 Generate .pkl Annotation Files

Generate .pkl files for each scenario:

```bash
# Generate .pkl files for Heidelberg
python tools/create_data_forainetv2.py forainetv2 \
    --root-path ./data/ForAINetV2_heidelberg \
    --out-dir ./data/ForAINetV2_heidelberg \
    --extra-tag forainetv2_heidelberg

# Generate .pkl files for UWaterloo
python tools/create_data_forainetv2.py forainetv2 \
    --root-path ./data/ForAINetV2_uwaterloo \
    --out-dir ./data/ForAINetV2_uwaterloo \
    --extra-tag forainetv2_uwaterloo

# Generate .pkl files for Combined
python tools/create_data_forainetv2.py forainetv2 \
    --root-path ./data/ForAINetV2_combined \
    --out-dir ./data/ForAINetV2_combined \
    --extra-tag forainetv2_combined
```

**Expected output:**
- `data/ForAINetV2_{scenario}/forainetv2_{scenario}_instance_data/*.npy` - Processed point clouds
- `data/ForAINetV2_{scenario}/forainetv2_{scenario}_oneformer3d_infos_train.pkl` - Training annotations
- `data/ForAINetV2_{scenario}/forainetv2_{scenario}_oneformer3d_infos_val.pkl` - Validation annotations

---

## ⚙️ Step 3: Configure Training

### 3.1 Config Files

Three config files are already created:
- `configs/oneformer3d_features_heidelberg.py`
- `configs/oneformer3d_features_uwaterloo.py`
- `configs/oneformer3d_features_combined.py`

Each inherits from the base config `configs/oneformer3d_qs_radius16_qp300_2many_features.py` and only changes the data paths.

**Key settings in base config:**
- `in_channels=6` (x, y, z + 3 scanner features)
- `num_semantic_classes=2` (wood, leaf)
- `load_dim=6`, `use_dim=[0, 1, 2, 3, 4, 5]`
- `stuff_classes=[]` (no ground)
- `thing_cls=[1, 2]` (wood, leaf)
- `val_evaluator` configured for 2-class evaluation

### 3.2 Training Hyperparameters

**Default settings:**
- Learning rate: `0.0001`
- Batch size: `2`
- Max epochs: `500`
- Validation interval: `50`
- Optimizer: `AdamW` with weight decay `0.05`

**Adjust if needed:**
- Reduce batch size if OOM: `batch_size=1`
- Adjust learning rate: `lr=0.00005` (lower) or `lr=0.0002` (higher)
- Reduce max points: `num_points=320000` (instead of 640000)

---

## 🚀 Step 4: Run Training

### 4.1 Set Environment

```bash
cd /home/rkaharly/ForestFormer3D
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH
```

**Important:** `PYTHONPATH` must be set so Python can find the `oneformer3d` module.

### 4.2 Train Model 1: Heidelberg-only

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_features_heidelberg.py \
    --work-dir work_dirs/heidelberg_features
```

### 4.3 Train Model 2: UWaterloo-only

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_features_uwaterloo.py \
    --work-dir work_dirs/uwaterloo_features
```

### 4.4 Train Model 3: Combined

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_features_combined.py \
    --work-dir work_dirs/combined_features
```

### 4.5 Monitor Training

**Tensorboard:**
```bash
# For Heidelberg
tensorboard --logdir=work_dirs/heidelberg_features/vis_data/ --host=0.0.0.0 --port=6006

# For UWaterloo
tensorboard --logdir=work_dirs/uwaterloo_features/vis_data/ --host=0.0.0.0 --port=6007

# For Combined
tensorboard --logdir=work_dirs/combined_features/vis_data/ --host=0.0.0.0 --port=6008
```

**Resume from checkpoint:**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_features_heidelberg.py \
    --work-dir work_dirs/heidelberg_features \
    --resume work_dirs/heidelberg_features/epoch_100.pth
```

**Output:**
- Checkpoints: `work_dirs/{scenario}_features/epoch_XXX.pth`
- Best model: `work_dirs/{scenario}_features/best.pth`
- Logs: `work_dirs/{scenario}_features/YYYYMMDD_HHMMSS.log`

---

## ✅ Step 5: Prepare Models for Inference

After training, you may need to fix the checkpoint for spconv compatibility (if you encounter loading errors):

```bash
# Fix Heidelberg model (if needed)
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/heidelberg_features/epoch_500.pth \
    --out-path work_dirs/heidelberg_features/epoch_500_fix.pth

# Fix UWaterloo model (if needed)
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/uwaterloo_features/epoch_500.pth \
    --out-path work_dirs/uwaterloo_features/epoch_500_fix.pth

# Fix Combined model (if needed)
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/combined_features/epoch_500.pth \
    --out-path work_dirs/combined_features/epoch_500_fix.pth
```

**Note:** If the model loads without errors, you can use the checkpoint directly without fixing.

---

## 🧪 Step 6: Evaluate Models

### 6.1 Test on Validation Set

```bash
# Test Heidelberg model
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_features_heidelberg.py \
    work_dirs/heidelberg_features/epoch_500.pth

# Test UWaterloo model
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_features_uwaterloo.py \
    work_dirs/uwaterloo_features/epoch_500.pth

# Test Combined model
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_features_combined.py \
    work_dirs/combined_features/epoch_500.pth
```

**Note:** Replace `epoch_500.pth` with `best.pth` if available, or use the latest epoch checkpoint.

### 6.2 Compare Results

Compare metrics across the three models:
- Semantic segmentation mIoU (wood vs. leaf)
- Per-class accuracy (wood precision/recall, leaf precision/recall)
- Instance segmentation accuracy (if applicable)

**Expected findings:**
- **Heidelberg-only**: Best performance on Heidelberg test data
- **UWaterloo-only**: Best performance on UWaterloo test data
- **Combined**: Better generalization, may perform well on both datasets

---

## 📊 Step 7: Inference on New Files

### 7.1 Using the Inference Script

The `infer_forestformer3d.py` script has been updated to support the scanner features model. It automatically detects the model type and uses the correct config.

**Basic usage:**
```bash
# Single file (PLY or LAS/LAZ) - auto-detects model and config
python infer_forestformer3d.py --input /path/to/file.ply

# Multiple files
python infer_forestformer3d.py --input file1.ply file2.las file3.ply

# Directory of files
python infer_forestformer3d.py --input /path/to/directory/

# Custom output directory
python infer_forestformer3d.py --input file.ply --output my_results/

# Specify model checkpoint explicitly
python infer_forestformer3d.py \
    --input file.ply \
    --model work_dirs/heidelberg_features/epoch_500.pth

# Specify config file explicitly (usually not needed - auto-detected)
python infer_forestformer3d.py \
    --input file.ply \
    --config configs/oneformer3d_features_heidelberg.py

# Use specific CUDA device
python infer_forestformer3d.py \
    --input file.ply \
    --cuda-device 0
```

**How it works:**
1. **Auto-detection**: The script automatically detects if you're using a scanner features model (checks for "features" or "heidelberg_features" in model path)
2. **Config selection**: Automatically uses the correct config file:
   - Scanner features model → `configs/oneformer3d_features_heidelberg.py` (or scenario-specific)
   - Legacy model → `configs/oneformer3d_qs_radius16_qp300_2many.py`
3. **Data preprocessing**: The data loader (`load_forainetv2_data.py`) automatically:
   - Extracts scanner features (intensity, return_number, number_of_returns) from PLY/LAS files
   - Normalizes them to [0, 1] range
   - Fills missing features with zeros (if not present in input file)

**Important:** 
- **Scanner features**: The model was trained with scanner features. Input files **should have** intensity, return_number, and number_of_returns for best performance.
- **Missing features**: If scanner features are missing, the data loader will fill them with zeros. This may degrade performance but won't crash.
- **File formats**: Supports PLY and LAS/LAZ files. LAS/LAZ files are automatically converted to PLY during preprocessing.
- **Output**: Prediction files are saved to `inference_runs/run_TIMESTAMP/` with naming: `{filename}_classified_by_ff3d.ply`
- **Output format**: The output PLY file contains:
  - Original coordinates and all original fields
  - `semantic_pred`: 0-indexed labels (0=wood, 1=leaf)
  - `instance_pred`: Instance IDs (if available)
  - `score`: Prediction confidence scores (if available)

### 7.2 Using MMDetection3D Test Script

You can also use the MMDetection3D test script directly for validation/test sets:

```bash
# Test on validation set
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_features_heidelberg.py \
    work_dirs/heidelberg_features/epoch_500.pth

# Test on custom test set (update test_dataloader in config first)
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_features_heidelberg.py \
    work_dirs/heidelberg_features/epoch_500.pth \
    --eval mIoU
```

### 7.3 Evaluation Script

The `evaluate_forestformer3d.py` script has been updated to support the new model's 0-indexed labels (0=wood, 1=leaf).

**Usage:**
```bash
# Evaluate on test files (auto-detects model type and runs inference if needed)
python evaluate_forestformer3d.py

# Specify model checkpoint
python evaluate_forestformer3d.py \
    --model-checkpoint work_dirs/heidelberg_features/epoch_500.pth

# Save classified point clouds
python evaluate_forestformer3d.py --save-classified-clouds

# Custom output directory
python evaluate_forestformer3d.py --output my_evaluation_results/

# Use specific CUDA device
python evaluate_forestformer3d.py --cuda-device 0
```

**How it works:**
1. **Auto-detection**: Automatically detects whether you're using the new model (0-indexed: 0=wood, 1=leaf) or legacy model (0,1=branch, 2=leaf) by checking the label range in predictions
2. **Label mapping**: 
   - New model: 0=wood → 0 (branch), 1=leaf → 1 (leaf)
   - Legacy model: 0,1=branch → 0, 2=leaf → 1
3. **Inference integration**: If prediction files don't exist, automatically runs inference first
4. **File matching**: Finds prediction files in `inference_runs/run_*/` directories (most recent run first)
5. **Evaluation files**: Uses files listed in `CONFIG.EVALUATION_FILES` in the script (update this list for your test files)

**Note:** Update `CONFIG.EVALUATION_FILES` in `evaluate_forestformer3d.py` to include your test files with ground truth labels.

### 7.4 Using the Model with Custom Input Data

**Requirements for input files:**
- **Format**: PLY or LAS/LAZ files
- **Coordinates**: Must have x, y, z coordinates
- **Scanner features** (recommended for best performance):
  - `intensity` (uint16: 0-65535)
  - `return_number` (uint8: 1-6)
  - `number_of_returns` (uint8: 1-7)
- **Missing features**: If scanner features are missing, they will be filled with zeros (may degrade performance)

**Quick start example:**
```bash
# Run inference on your custom PLY file
python infer_forestformer3d.py --input /path/to/your/file.ply

# Run inference on LAS file
python infer_forestformer3d.py --input /path/to/your/file.las

# Run inference on multiple files
python infer_forestformer3d.py --input file1.ply file2.las file3.ply

# Output will be in: inference_runs/run_TIMESTAMP/
```

**What happens during inference:**
1. Input file is copied to `data/ForAINetV2/test_data/`
2. If LAS/LAZ, it's converted to PLY (preserving all fields including scanner features)
3. Data is preprocessed using `load_forainetv2_data.py`:
   - Extracts coordinates (x, y, z)
   - Extracts and normalizes scanner features (if present)
   - Fills missing scanner features with zeros
   - Creates 6-channel input: [x, y, z, intensity, return_number, number_of_returns]
4. Model runs inference
5. Predictions are matched back to original point cloud coordinates
6. Output saved as `{filename}_classified_by_ff3d.ply` with:
   - Original coordinates and all original fields
   - `semantic_pred`: 0=wood, 1=leaf (0-indexed)
   - `instance_pred`: Instance IDs
   - `score`: Confidence scores

**Verifying your input files:**
```bash
# Check if PLY file has scanner features
python -c "
from plyfile import PlyData
import numpy as np
ply = PlyData.read('your_file.ply')
vertex = ply['vertex'].data
print('Fields:', vertex.dtype.names)
if 'intensity' in vertex.dtype.names:
    print('Intensity range:', vertex['intensity'].min(), '-', vertex['intensity'].max())
if 'return_number' in vertex.dtype.names:
    print('Return number range:', vertex['return_number'].min(), '-', vertex['return_number'].max())
if 'number_of_returns' in vertex.dtype.names:
    print('Number of returns range:', vertex['number_of_returns'].min(), '-', vertex['number_of_returns'].max())
"
```

---

## 🔧 Implementation Details & Code Modifications

This section documents all the changes made to enable training from scratch with scanner features.

### Scanner Features Used

1. **intensity** (uint16: 0-65535)
   - Normalized: `intensity / 65535.0` → [0, 1]
   - Reflects material properties (wood vs. leaf)

2. **return_number** (uint8: 1-6)
   - Normalized: `(return_number - 1) / 5.0` → [0, 1]
   - First return often hits leaves, later returns hit wood

3. **number_of_returns** (uint8: 1-7)
   - Normalized: `(number_of_returns - 1) / 6.0` → [0, 1]
   - Indicates pulse penetration depth

### 1. Modified Data Loading (`data/ForAINetV2/load_forainetv2_data.py`)

**Problem:** Original script only extracted x, y, z coordinates, discarding scanner features.

**Fix (lines 142-169):**
```python
# Extract and normalize scanner features (intensity, return_number, number_of_returns)
scanner_features = []

# Intensity (normalize to 0-1)
if 'intensity' in pcd.dtype.names:
    intensity = pcd['intensity'].astype(np.float32)
    intensity_norm = intensity / 65535.0
    scanner_features.append(intensity_norm)

# Return number (normalize to 0-1)
if 'return_number' in pcd.dtype.names:
    return_num = pcd['return_number'].astype(np.float32)
    return_num_norm = (return_num - 1.0) / 5.0
    scanner_features.append(return_num_norm)

# Number of returns (normalize to 0-1)
if 'number_of_returns' in pcd.dtype.names:
    num_returns = pcd['number_of_returns'].astype(np.float32)
    num_returns_norm = (num_returns - 1.0) / 6.0
    scanner_features.append(num_returns_norm)

# Concatenate coordinates with scanner features
if len(scanner_features) == 3:
    points = np.hstack([points, np.column_stack(scanner_features)])
    num_features = 6  # x, y, z + 3 scanner features
else:
    num_features = 3  # Fallback to coordinates only
```

**Result:** Points array now has shape (N, 6) instead of (N, 3).

### 2. Updated Semantic Label Processing (`data/ForAINetV2/load_forainetv2_data.py`)

**Problem:** Original script converted labels from 0-indexed to 1-indexed, adding a ground class.

**Fix (lines 186-203):**
```python
# Our data: semantic_seg = [1, 2] (1=wood, 2=leaf, no ground)
# Keep as-is (1-indexed), no conversion needed
if "semantic_seg" in pcd.dtype.names:
    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    # Keep 1-indexed: 1=wood, 2=leaf (no ground class)
else:
    # Fallback: all wood (class 1)
    semantic_seg = np.ones((points.shape[0],), dtype=np.int64)

# No ground class in our data
bg_sem = np.array([])  # Empty array = no background/ground class
label_ids = semantic_seg  # Keep 1-indexed: 1=wood, 2=leaf
```

**Result:** Labels remain 1-indexed (1=wood, 2=leaf) without adding ground class.

### 3. Created New Config Files

**Files created:**
- `configs/oneformer3d_qs_radius16_qp300_2many_features.py` (base config)
- `configs/oneformer3d_features_heidelberg.py`
- `configs/oneformer3d_features_uwaterloo.py`
- `configs/oneformer3d_features_combined.py`

**Key changes in base config:**

**Model settings:**
```python
in_channels=6,  # Changed from 3: x, y, z, intensity, return_number, number_of_returns
num_semantic_classes=2,  # Changed from 3: only wood (1) and leaf (2), no ground
stuff_classes=[],  # Changed from [0]: no ground/stuff class
thing_cls=[1, 2],  # wood (1), leaf (2)
```

**Data pipeline:**
```python
load_dim=6,  # Changed from 3
use_dim=[0, 1, 2, 3, 4, 5],  # Changed from [0, 1, 2]: use all 6 dimensions
```

**Transform parameters:**
```python
dict(type='AddSuperPointAnnotations', num_classes=num_semantic_classes, stuff_classes=[]),
dict(type='PointInstClassMapping_', num_classes=num_instance_classes),
```

**Evaluator settings:**
```python
# Note: Labels are 1-indexed: 1=wood, 2=leaf
class_names = ['wood', 'leaf']  # Only 2 classes (no ground)
label2cat = {1: 'wood', 2: 'leaf'}  # Map 1-indexed labels to class names
sem_mapping = [1, 2]  # 1=wood, 2=leaf (1-indexed, no 0/ground)
inst_mapping = sem_mapping  # Same mapping for instances
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[],  # No stuff/ground class
    thing_class_inds=[1, 2],  # wood (1), leaf (2)
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
```

### 4. Fixed GridSample Transform (`oneformer3d/transforms_3d.py`)

**Problem:** `AddSuperPointAnnotations` transform requires `sp_pts_mask` (superpoint mask), but `GridSample` didn't create it. This caused `KeyError: 'sp_pts_mask'` during training.

**Fix (lines 717-718):**
```python
# Create sp_pts_mask from grid hash if it doesn't exist
# This assigns each point to a superpoint based on its grid cell
if 'sp_pts_mask' not in input_dict:
    # Use the inverse indices from unique grid cells as superpoint IDs
    # This creates a superpoint for each unique grid cell
    input_dict['sp_pts_mask'] = inverse.numpy()
```

**Result:** Superpoint masks are automatically generated from grid cells, enabling the training pipeline to work.

### 5. Fixed .pkl File Format (`tools/create_data_forainetv2.py` and manual fix)

**Problem:** Generated .pkl files had incorrect format. The dataset loader expects a dict with `metainfo` and `data_list`, where each item in `data_list` has a `lidar_points` structure.

**Original format (incorrect):**
```python
# List of dicts with 'point_cloud', 'pts_path', etc.
[
    {'point_cloud': {...}, 'pts_path': '...', ...},
    ...
]
```

**Required format (correct):**
```python
{
    'metainfo': {
        'categories': {'wood': 1, 'leaf': 2},
        'dataset': 'forainetv2_heidelberg',
        'info_version': '1.1'
    },
    'data_list': [
        {
            'lidar_points': {
                'num_pts_feats': 6,
                'lidar_path': 'file.bin'
            },
            'pts_semantic_mask_path': 'file.bin',
            'pts_instance_mask_path': 'file.bin',
            'instances': [...],
            'axis_align_matrix': [...]
        },
        ...
    ]
}
```

**Fix:** Modified the .pkl generation to create the correct structure with `lidar_points` nested dict.

### 6. Fixed Dataset Config Parameters

**Problem:** `ForAINetV2SegDataset_` doesn't accept `num_classes` parameter, causing `TypeError: BaseDataset.__init__() got an unexpected keyword argument 'num_classes'`.

**Fix:** Removed `num_classes` from all dataset configs:
```python
# Before (incorrect):
dataset=dict(
    type=dataset_type,
    ...
    num_classes=num_instance_classes),  # ❌ Not accepted

# After (correct):
dataset=dict(
    type=dataset_type,
    ...
    # num_classes removed ✅
)
```

### 7. Fixed Transform Parameters

**Problem:** `AddSuperPointAnnotations` and `PointInstClassMapping_` transforms require specific parameters that weren't provided in the config.

**Fix for AddSuperPointAnnotations:**
```python
# Before (incorrect):
dict(type='AddSuperPointAnnotations'),  # ❌ Missing required args

# After (correct):
dict(type='AddSuperPointAnnotations', num_classes=num_semantic_classes, stuff_classes=[]),  # ✅
```

**Fix for PointInstClassMapping_:**
```python
# Before (incorrect):
dict(type='PointInstClassMapping_'),  # ❌ Missing required arg

# After (correct):
dict(type='PointInstClassMapping_', num_classes=num_instance_classes),  # ✅
```

### 8. Fixed Instance Data Directory Detection (`tools/forainetv2_data_utils.py`)

**Problem:** Script hardcoded `forainetv2_instance_data` directory name, causing `FileNotFoundError` when looking for `.npy` files in scenario-specific directories (e.g., `forainetv2_heidelberg_instance_data`).

**Fix:**
```python
# Before (incorrect):
self.instance_data_dir = osp.join(self.root_dir, 'forainetv2_instance_data')  # ❌ Hardcoded

# After (correct):
# Auto-detect instance_data directory name from root_dir and extra_tag
info_prefix = osp.basename(self.root_dir.rstrip('/')).replace('ForAINetV2_', 'forainetv2_')
if info_prefix == 'forainetv2':
    info_prefix = 'forainetv2'  # Default case
self.instance_data_dir = osp.join(self.root_dir, f'{info_prefix}_instance_data')  # ✅ Dynamic
```

**Result:** Script correctly locates `.npy` files for each scenario (heidelberg, uwaterloo, combined).

---

## 🐛 Challenges & Problems Encountered

### Challenge 1: Missing val_evaluator

**Error:**
```
ValueError: val_dataloader, val_cfg, and val_evaluator should be either all None or not None
```

**Root cause:** MMEngine requires that if `val_dataloader` and `val_cfg` are defined, `val_evaluator` must also be defined.

**Solution:** Added `val_evaluator` configuration with proper class mappings for 2-class segmentation (wood, leaf).

### Challenge 2: Dataset Parameter Mismatch

**Error:**
```
TypeError: BaseDataset.__init__() got an unexpected keyword argument 'num_classes'
```

**Root cause:** The `ForAINetV2SegDataset_` class inherits from `ScanNetDataset`, which doesn't accept `num_classes` as a parameter.

**Solution:** Removed `num_classes` from all dataset configurations. The number of classes is determined from the model config, not the dataset config.

### Challenge 3: Missing Transform Parameters

**Error:**
```
TypeError: AddSuperPointAnnotations.__init__() missing 2 required positional arguments: 'num_classes' and 'stuff_classes'
```

**Root cause:** The transform requires these parameters to properly handle semantic and instance masks, but they weren't specified in the config.

**Solution:** Added `num_classes=num_semantic_classes` and `stuff_classes=[]` to all `AddSuperPointAnnotations` transforms in the pipeline.

### Challenge 4: Incorrect .pkl File Format

**Error:**
```
TypeError: The annotations loaded from annotation file should be a dict, but got <class 'list'>!
```

**Root cause:** The .pkl files were generated as lists instead of dicts with `metainfo` and `data_list` structure.

**Solution:** Modified the .pkl generation to wrap data in the correct structure with `metainfo` and `data_list` keys, and ensure each item has a `lidar_points` nested dict.

### Challenge 5: Missing sp_pts_mask

**Error:**
```
KeyError: 'sp_pts_mask'
```

**Root cause:** The `AddSuperPointAnnotations` transform requires `sp_pts_mask` (superpoint mask) to aggregate instance and semantic masks, but `GridSample` transform didn't create it.

**Solution:** Modified `GridSample` to automatically create `sp_pts_mask` from grid cell hashes if it doesn't exist. This assigns each point to a superpoint based on its grid cell location.

### Challenge 6: Instance Data Directory Path Issues

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data/ForAINetV2_heidelberg/forainetv2_instance_data/...'
```

**Root cause:** The script hardcoded `forainetv2_instance_data` but the actual directory was `forainetv2_heidelberg_instance_data`.

**Solution:** Modified `forainetv2_data_utils.py` to dynamically determine the instance data directory name based on the `root_dir` and `extra_tag` parameters.

### Challenge 7: Environment Setup Issues

**Error:**
```
ImportError: /fsys1/home/rkaharly/venv/lib/python3.10/site-packages/mmcv/_ext.cpython-310-x86_64-linux-gnu.so: undefined symbol
```

**Root cause:** Wrong Python environment (system venv instead of conda environment).

**Solution:** Ensured proper conda environment activation and `PYTHONPATH` setup in all scripts:
```bash
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH
```

---

## 🐛 Troubleshooting

### "Cannot find valid image after 1000!"

**Cause:** Data loading issues or empty scenes.

**Solution:**
1. Verify PLY files have correct fields
2. Check that `.npy` files were generated correctly
3. Verify `train_list.txt` and `val_list.txt` contain valid file names

### Out of Memory

**Solutions:**
- Reduce `batch_size` in config (2 → 1)
- Reduce `num_points` in `PointSample_` (640000 → 320000)
- Reduce `radius` in config (16 → 12)

### "KeyError: 'epoch'"

**Cause:** Epoch not passed to loss function.

**Solution:** Already fixed in code. Uses `kwargs.get('epoch', 0)`.

### Scanner Features Not Loading

**Cause:** PLY files missing scanner features.

**Solution:**
1. Verify standardized files have `intensity`, `return_number`, `number_of_returns`
2. Check data loading script is using the updated version
3. Re-run data preprocessing

### Model Performance Issues

**If model doesn't converge:**
- Check learning rate (try lower: 0.00005)
- Verify feature normalization (should be in [0, 1])
- Check data quality (labels correct, features present)

**If overfitting:**
- Increase validation interval
- Add data augmentation
- Reduce model capacity

### Dataset shows 0 instances for wood/leaf

**Cause:** Statistics count bounding boxes (instances), not semantic points.

**Solution:** This is normal. The model still trains on semantic segmentation. Verify actual data:
```bash
python -c "
import numpy as np
sem = np.load('data/ForAINetV2_heidelberg/forainetv2_heidelberg_instance_data/FILE_sem_label.npy')
print('Semantic classes:', np.unique(sem))
print('Counts:', dict(zip(*np.unique(sem, return_counts=True))))
"
```

---

## 📝 Summary Checklist

### Before Training:
- [ ] Standardized data files ready in `data/standardized/`
- [ ] Run `prepare_training_splits.py` for all scenarios
- [ ] Verify train/val splits created correctly
- [ ] Run data preprocessing for each scenario
- [ ] Generate `.pkl` annotation files
- [ ] Verify config files exist for each scenario
- [ ] Set `PYTHONPATH` environment variable

### During Training:
- [ ] Monitor training with Tensorboard
- [ ] Check validation metrics
- [ ] Save checkpoints regularly
- [ ] Note training time and GPU usage

### After Training:
- [ ] Fix checkpoints with `fix_spconv_checkpoint.py`
- [ ] Evaluate on validation set
- [ ] Compare results across three models
- [ ] Document findings and conclusions

---

## 📚 Additional Resources

- **Base config**: `configs/oneformer3d_qs_radius16_qp300_2many_features.py`
- **Data loading**: `data/ForAINetV2/load_forainetv2_data.py`
- **Finetuning guide**: `FINETUNING_GUIDE.md` (for reference on coordinate-only training)

---

## 🎯 Expected Training Times

**Rough estimates** (depends on hardware):
- **Heidelberg**: ~10-15 hours (10 files, ~50M points)
- **UWaterloo**: ~5-8 hours (17 files, ~13M points)
- **Combined**: ~15-20 hours (27 files, ~63M points)

**GPU requirements:**
- Minimum: 16GB VRAM (batch_size=1)
- Recommended: 24GB+ VRAM (batch_size=2)

---

Good luck with training! 🚀
