# ForestFormer3D

Deep learning model for LiDAR tree point cloud segmentation (branches/leaves).

## Setup

### Environment
```bash
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
cd /home/rkaharly/ForestFormer3D
```

### Required Files
- Model checkpoint: `work_dirs/clean_forestformer/epoch_3000_fix.pth`
- Config: `configs/oneformer3d_qs_radius16_qp300_2many.py`
- Test data: Place files in `data/ForAINetV2/test_data/` or use `--input` flag

## Quick Start

### Run Inference
```bash
# Single file (model path required)
python infer_forestformer3d.py --input test_files/256_leafon.ply --model work_dirs/heidelberg_features/epoch_500.pth

# Multiple files or directory
python infer_forestformer3d.py --input file1.ply file2.las test_files/ --model work_dirs/heidelberg_features/epoch_500.pth
```

**Note:** The `--model` argument is **required**. Specify the path to your trained model checkpoint.

Output: `inference_runs/run_TIMESTAMP/<filename>_classified_by_ff3d.ply`

### Run Evaluation
1. Edit `evaluate_forestformer3d.py`, add files to `CONFIG.EVALUATION_FILES`:
```python
EVALUATION_FILES = [
    "test_files/256_leafon.ply",
    "test_files/108_leafon.ply",
]
```

2. Run:
```bash
python evaluate_forestformer3d.py --save-classified-clouds
```

Output: Metrics logged to W&B, report in `evaluation_results_forestformer3d/`

## Scripts

### `infer_forestformer3d.py` - Inference

Run inference on PLY/LAS/LAZ files.

**Usage:**
```bash
python infer_forestformer3d.py --input <file_or_dir> [options]
```

**Options:**
- `--input`: Input file(s) or directory (required)
- `--model`: Model checkpoint path (REQUIRED)
- `--output`: Output directory (default: `inference_runs/run_TIMESTAMP`)
- `--config`: Config file path (default: auto-detected based on model)
- `--cuda-device`: GPU ID (default: 0)
- `--keep-intermediate-files`: Keep intermediate files (default: False)
- `--verbose`: Detailed logging

**Examples:**
```bash
python infer_forestformer3d.py --input test_files/256_leafon.las --model work_dirs/heidelberg_features/epoch_500.pth
python infer_forestformer3d.py --input file1.ply file2.las --model work_dirs/heidelberg_features/epoch_500.pth --output my_results/
```

### `evaluate_forestformer3d.py` - Evaluation

Evaluate predictions against ground truth and log to W&B.

**Usage:**
```bash
python evaluate_forestformer3d.py [options]
```

**Options:**
- `--save-classified-clouds`: Save classified point clouds
- `--output`: Output directory (default: `evaluation_results_forestformer3d/`)
- `--model-checkpoint`: Model checkpoint path (default: auto-detect)
- `--verbose`: Detailed logging

**Configuration:**
Edit `CONFIG` class in the script:
- `EVALUATION_FILES`: List of ground truth file paths
- `USE_FILTERED_OUTPUT`: Use filtered predictions (default: False)
- `PREDICTION_OUTPUT_DIR`: Directory with predictions (default: `work_dirs/output/round_1`)
- `SPATIAL_MATCH_TOLERANCE`: Point matching tolerance in meters (default: 0.01)

**Examples:**
```bash
# Edit EVALUATION_FILES in script first, then:
python evaluate_forestformer3d.py
python evaluate_forestformer3d.py --save-classified-clouds --verbose
```

### `tools/convert_las_to_ply.py` - File Conversion

Convert LAS/LAZ to PLY preserving all fields.

**Usage:**
```bash
python tools/convert_las_to_ply.py input.las output.ply
```

## File Formats

**Input (Inference):**
- `.ply`, `.las`, `.laz`
- Files are automatically standardized (field names normalized) with caching
- Cache location: `work_dirs/standardized_cache/`
- Cache is automatically invalidated when source files change

**Output (Inference):**
- `.ply` with `semantic_pred` field:
  - `0` = wood (for scanner features model)
  - `1` = leaf (for scanner features model)
  - Legacy model: `0` or `1` = branch, `2` = leaf

**Ground Truth (Evaluation):**
- `.ply` with `scalar_Classification` or `classification` field (0=branch, 1=leaf)
- `.las`/`.laz` with `classification` field (0=branch, 1=leaf)

## Output Locations

**Inference:**
- Final predictions: `inference_runs/run_TIMESTAMP/<filename>_classified_by_ff3d.ply`
- Intermediate files: `work_dirs/output/` (cleaned up by default)

**Standardization Cache:**
- Location: `work_dirs/standardized_cache/`
- Format: `{filename}_{hash}.ply`
- Automatically created and reused for faster subsequent runs
- Clear cache: `rm -rf work_dirs/standardized_cache/`

**Evaluation:**
- Report: `evaluation_results_forestformer3d/evaluation_TIMESTAMP/evaluation_report.txt`
- Classified clouds: Same directory (if `--save-classified-clouds`)
- W&B: Project "tree-segmentation"

## W&B Metrics

Logged metrics include:
- Overall: accuracy, precision, recall, F1, Jaccard
- Per-class: branch/leaf precision, recall, F1, Jaccard
- Height-stratified: F1 scores for lower/middle/upper tree sections
- Confusion matrix: TP, FP, TN, FN counts

## Troubleshooting

**LAS conversion fails:**
- Ensure `laspy` is installed: `pip install laspy`

**CUDA out of memory:**
- Use different GPU: `--cuda-device 1`
- Reduce batch size in config

**No prediction file found:**
- Run inference first
- Check file names match (ground truth: `file.ply` → prediction: `file_round1.ply`)

**Points don't match in evaluation:**
- Normal if coordinates differ slightly
- Adjust `SPATIAL_MATCH_TOLERANCE` in `CONFIG` (default: 0.01m)

## Notes

- **Model path is required** for inference (no default model)
- Files are automatically standardized with caching for consistent field names
- Model expects PLY format; LAS/LAZ are auto-converted
- Evaluation uses spatial matching (KDTree) to align prediction and ground truth points
- Standardization cache speeds up repeated runs on the same files

