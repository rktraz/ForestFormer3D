# ForestFormer3D Finetuning Guide

## 📋 Data Requirements

### Input Data Format

Your input data must be in **PLY format** with the following structure:

**Required fields:**
- `x`, `y`, `z` (float64): Point coordinates
- `semantic_seg` (int32): Semantic labels
  - `0` = ground
  - `1` = wood  
  - `2` = leaf
- `treeID` (int32): Instance IDs
  - `0` = background/ground points
  - `1+` = tree instance IDs (can be same for all points if single tree per file)

**Data type:** Binary PLY format

**Note:** If your data doesn't have ground points, you must add dummy ground points (see Step 1).

---

## 📁 Step 1: Data Preparation

### 1.1 Convert Your Data to Training Format

If your data is in LAS/LAZ/PLY format, convert it using the preparation script:

```bash
python tools/prepare_my_custom_training_data.py \
    --input-dir /path/to/your/raw/data \
    --output-dir data/ForAINetV2 \
    --subdir train_val_data \
    --num-ground-points 1000 \
    --ground-height-offset -0.5
```

**What this script does:**
- Converts LAS/LAZ files to PLY format (if needed)
- Maps classification labels: `0` (wood) → `1`, `1` (leaf) → `2`
- Adds dummy ground points (semantic_seg=0, treeID=0) at the bottom of each point cloud
- Sets treeID=1 for all non-ground points
- Removes unnecessary fields, keeps only: `x`, `y`, `z`, `semantic_seg`, `treeID`
- Saves binary PLY files ready for training

**Parameters:**
- `--input-dir`: Directory containing your LAS/LAZ/PLY files
- `--output-dir`: Base output directory (creates `train_val_data/` or `test_data/` subdirectory)
- `--subdir`: Subdirectory name - `train_val_data` (default) or `test_data`
- `--num-ground-points`: Number of ground points to add per file (default: 1000)
- `--ground-height-offset`: Z offset for ground points relative to min Z (default: -0.5)

**Output format:**
- PLY files with fields: `x`, `y`, `z` (float64), `semantic_seg` (int32: 0=ground, 1=wood, 2=leaf), `treeID` (int32: 0=ground, 1+=tree)

### 1.2 Organize Data Structure

After conversion, organize your files:

```
data/ForAINetV2/
├── train_val_data/     # All training and validation PLY files
│   ├── file1.ply
│   ├── file2.ply
│   └── ...
├── test_data/          # Optional: Test PLY files
│   └── ...
└── meta_data/
    ├── train_list.txt  # Base names only (no .ply extension)
    ├── val_list.txt
    └── test_list.txt
```

**train_list.txt example:**
```
file1
file2
file3
```

### 1.3 Run Data Preprocessing

**Note:** If you have existing PLY files that were created with the old script (labels 2,3 instead of 1,2), you can use `fix_ply_labels_and_add_ground.py` to fix them. Otherwise, use `prepare_my_custom_training_data.py` which now does everything correctly.

```bash
cd data/ForAINetV2

# Delete old .bin files if regenerating
rm -rf forainetv2_instance_data/*

# Generate .bin files from PLY files
python batch_load_ForAINetV2_data.py \
    --train_scan_names_file meta_data/train_list.txt \
    --val_scan_names_file meta_data/val_list.txt \
    --test_scan_names_file meta_data/test_list.txt

# Return to project root
cd ../..

# Generate .pkl annotation files
python tools/create_data_forainetv2.py forainetv2
```

**Output:**
- `data/ForAINetV2/forainetv2_instance_data/*.npy` - Processed point clouds
- `data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl` - Training annotations
- `data/ForAINetV2/forainetv2_oneformer3d_infos_val.pkl` - Validation annotations
- `data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl` - Test annotations

---

## ⚙️ Step 2: Configure Training

Edit `configs/oneformer3d_qs_radius16_qp300_2many.py`:

1. **Set pretrained model path:**
   ```python
   load_from = 'work_dirs/clean_forestformer/epoch_3000_fix.pth'
   ```

2. **Adjust learning rate (recommended for finetuning):**
   ```python
   optim_wrapper = dict(
       type='OptimWrapper',
       optimizer=dict(type='AdamW', lr=1e-05, weight_decay=0.05),
       clip_grad=dict(max_norm=10, norm_type=2))
   ```

3. **Training epochs:**
   ```python
   train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=50)
   ```

---

## 🚀 Step 3: Run Training

```bash
cd /home/rkaharly/ForestFormer3D
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_qs_radius16_qp300_2many.py \
    --work-dir work_dirs/my_finetuned_model
```

**Important:** `PYTHONPATH` must be set so Python can find the `oneformer3d` module.

**Output:**
- Checkpoints saved to `work_dirs/my_finetuned_model/`
- Best model: `best.pth`
- Latest epoch: `epoch_XXX.pth`
- Logs: `work_dirs/my_finetuned_model/YYYYMMDD_HHMMSS.log`

### Monitor Training

**Tensorboard:**
```bash
tensorboard --logdir=work_dirs/my_finetuned_model/vis_data/ --host=0.0.0.0 --port=6006
```

**Resume from checkpoint:**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/oneformer3d_qs_radius16_qp300_2many.py \
    --work-dir work_dirs/my_finetuned_model \
    --resume work_dirs/my_finetuned_model/epoch_100.pth
```

---

## ✅ Step 4: Prepare Model for Inference

After training, fix the checkpoint:

```bash
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/my_finetuned_model/best.pth \
    --out-path work_dirs/my_finetuned_model/best_fix.pth
```

---

## 🧪 Step 5: Use the Model

### Test on validation/test set:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/oneformer3d_qs_radius16_qp300_2many.py \
    work_dirs/my_finetuned_model/best_fix.pth
```

### Inference on new files:
```bash
python infer_forestformer3d.py \
    --input /path/to/test/file.ply \
    --model work_dirs/my_finetuned_model/best_fix.pth \
    --output my_results/
```

---

## 🔧 Code Modifications Made

The following code changes were required to enable finetuning on custom data:

### 1. Fixed Data Loading (`data/ForAINetV2/load_forainetv2_data.py`)

**Problem:** Script used default values instead of reading from PLY files.

**Fix (lines 155-169):**
```python
# Read semantic_seg and treeID from PLY file
# PLY files use 0-indexed: 0=ground, 1=wood, 2=leaf
# Script expects 1-indexed: 1=ground, 2=wood, 3=leaf
if "semantic_seg" in pcd.dtype.names:
    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    semantic_seg = semantic_seg + 1  # Convert 0-indexed to 1-indexed
else:
    semantic_seg = np.ones((points.shape[0],), dtype=np.int64)

if "treeID" in pcd.dtype.names:
    treeID = pcd["treeID"].astype(np.int64)
else:
    treeID = np.zeros((points.shape[0],), dtype=np.int64)
```

### 2. Fixed SkipEmptyScene_ Transform (`oneformer3d/transforms_3d.py`)

**Problem:** Required 2+ unique instance IDs, filtering out single-tree scenes.

**Fix (lines 447-468):**
```python
def transform(self, input_dict):
    # Only skip if truly empty (no points)
    if len(input_dict["points"]) == 0:
        return None
    
    # Allow any scene with points (for wood/leaf segmentation)
    return input_dict
```

### 3. Fixed Instance Mask Handling (`oneformer3d/oneformer3d.py`)

**Problem:** Empty instance mask caused `one_hot()` to fail.

**Fix (lines 3145-3147):**
```python
_, instance_mask_clone = torch.unique(instance_mask_clone, return_inverse=True, sorted=True)
# Handle empty instance mask
if instance_mask_clone.numel() == 0 or instance_mask_clone.max() < 0:
    instance_mask_clone = torch.zeros(pts_instance_mask.shape[0], dtype=pts_instance_mask.dtype, device=pts_instance_mask.device)
num_classes = int(instance_mask_clone.max().item()) + 1 if instance_mask_clone.numel() > 0 else 1
one_hot_inst_mask = torch.nn.functional.one_hot(instance_mask_clone, num_classes=num_classes)
```

### 4. Fixed Epoch Access (`oneformer3d/oneformer3d.py`)

**Problem:** `kwargs['epoch']` not always available, causing KeyError.

**Fix (line 1972):**
```python
# Changed from: if kwargs['epoch'] > self.prepare_epoch:
# To:
current_epoch = kwargs.get('epoch', 0)
if current_epoch > self.prepare_epoch:
```

### 5. Fixed .pkl File Generation (`tools/update_infos_to_v2.py`)

**Problem:** METAINFO classes defined as string instead of tuple.

**Fix (line 391):**
```python
# Changed from: ('tree')
# To:
('tree',)  # Comma makes it a tuple
```

### 6. Removed Unused Import (`data/ForAINetV2/batch_load_ForAINetV2_data.py`)

**Problem:** `segmentator` module not available but imported.

**Fix (line 19):**
```python
# import segmentator  # Commented out - not used
```

### 7. Disabled SkipEmptyScene_ in Config (`configs/oneformer3d_qs_radius16_qp300_2many.py`)

**Problem:** Transform was filtering out valid scenes.

**Fix (line 105):**
```python
# dict(type='SkipEmptyScene_'),  # Commented out - allowing all scenes
```

### 8. Disabled filter_empty_gt (`configs/oneformer3d_qs_radius16_qp300_2many.py`)

**Problem:** Filter was removing single-instance scenes.

**Fix (line 179):**
```python
filter_empty_gt=False,  # Disabled to allow single-instance scenes
```

---

## 📊 Data Format Details

### PLY File Structure

**Binary PLY format with structured array:**
```python
dtype = [
    ('x', 'f8'),           # float64
    ('y', 'f8'),          # float64
    ('z', 'f8'),          # float64
    ('semantic_seg', 'i4'),  # int32: 0=ground, 1=wood, 2=leaf
    ('treeID', 'i4')      # int32: 0=background, 1+=instance ID
]
```

### Label Mapping

**Input (your PLY files):**
- `semantic_seg`: 0-indexed (0=ground, 1=wood, 2=leaf)

**After preprocessing:**
- Converted to 1-indexed (1=ground, 2=wood, 3=leaf) for internal processing
- Then converted back to 0-indexed (0=ground, 1=wood, 2=leaf) for model

### Instance IDs

- `treeID=0`: Background/ground points (not used for instance segmentation)
- `treeID=1+`: Valid instance IDs
- For single-tree scenes: All non-ground points can have `treeID=1`

---

## 🐛 Troubleshooting

### Dataset shows 0 instances for wood/leaf

**Cause:** Statistics count bounding boxes (instances), not semantic points.

**Solution:** This is normal. Verify actual data:
```bash
python -c "
import numpy as np
sem = np.load('data/ForAINetV2/forainetv2_instance_data/107_labeled_sem_label.npy')
print('Semantic classes:', np.unique(sem))
print('Counts:', dict(zip(*np.unique(sem, return_counts=True))))
"
```

### "Cannot find valid image after 1000!"

**Cause:** `SkipEmptyScene_` or other transforms filtering out all scenes.

**Solution:** Already fixed in code. If persists:
1. Check that PLY files have correct fields
2. Verify `.bin` files were regenerated after fixing PLY files
3. Check that semantic labels are 0, 1, 2 (not 2, 3)

### "KeyError: 'epoch'"

**Cause:** Epoch not passed to loss function.

**Solution:** Already fixed. Uses `kwargs.get('epoch', 0)`.

### Out of Memory

- Reduce `radius` in config (16 → 12)
- Reduce `batch_size` (2 → 1)
- Reduce `num_points` in `PointSample_` (640000 → 320000)

---

## 📝 Summary Checklist

- [ ] Convert LAS/LAZ/PLY files using `prepare_my_custom_training_data.py`
- [ ] PLY files have correct format: `x`, `y`, `z`, `semantic_seg` (0,1,2), `treeID`
- [ ] Create `train_list.txt`, `val_list.txt` in `meta_data/`
- [ ] Run `batch_load_ForAINetV2_data.py` to generate .bin files
- [ ] Run `create_data_forainetv2.py` to generate .pkl files
- [ ] Set `load_from` in config to pretrained model
- [ ] Set `PYTHONPATH` before training
- [ ] Run training command
- [ ] Fix checkpoint after training
- [ ] Test on validation/test set
