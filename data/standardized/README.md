# Standardized Point Cloud Data

This directory contains standardized point cloud files from two datasets:
- **Heidelberg**: 10 files, ~50M points
- **UWaterloo**: 17 files, ~13M points

## Standardization Process

All files have been standardized to have:
- **Consistent field names**: No more `scalar_` prefixes, all lowercase
- **Consistent data types**: 
  - Coordinates: `float64`
  - Intensity: `uint16`
  - Return numbers: `uint8`
  - Classification: `uint8`
  - RGB colors: `uint16` (if present)
  - Labels: `int32`

## Common Fields (All Files)

- `x`, `y`, `z` (float64): Coordinates
- `intensity` (uint16): LiDAR intensity
- `return_number` (uint8): Return number (1-6)
- `number_of_returns` (uint8): Number of returns (1-7)
- `classification` (uint8): 0=wood, 1=leaf
- `user_data` (uint8): User data field
- `point_source_id` (uint16): Point source ID
- `gps_time` (float64): GPS time (if valid)
- `semantic_seg` (int32): Semantic label (0=ground, 1=wood, 2=leaf)
- `treeID` (int32): Tree instance ID

## Dataset-Specific Fields

### Heidelberg
- Additional: `gps_time` (always present)

### UWaterloo
- Additional: `scan_angle` (int16), `red`, `green`, `blue` (uint16)

## File Structure

```
data/standardized/
├── heidelberg/
│   ├── AcePse_SP02_04_2019-08-23_q2_TLS-on_c.ply
│   ├── FagSyl_BR01_01_2019-07-08_q1_TLS-on_c.ply
│   └── ... (10 files total)
└── uwaterloo/
    ├── 107_labeled.ply
    ├── 129_labeled.ply
    └── ... (17 files total)
```

## Usage

These standardized files can now be used for:
1. Training models with scanner features
2. Consistent data loading pipeline
3. Feature extraction and analysis

## Standardization Script

Created: `tools/standardize_pointcloud_data.py`

Usage:
```bash
python tools/standardize_pointcloud_data.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --preserve-labels
```

## Statistics

### Heidelberg Dataset
- Total files: 10
- Total points: 49,908,425
- Average points per file: ~5M
- Fields per file: 12 (coordinates + scanner features + labels)

### UWaterloo Dataset
- Total files: 17
- Total points: 13,365,545
- Average points per file: ~786K
- Fields per file: 16 (coordinates + scanner features + RGB + labels)

## Next Steps

1. ✅ Data standardization complete
2. ⏭️ Create data loading pipeline for scanner features
3. ⏭️ Extend model architecture to use additional features
4. ⏭️ Train new model with scanner features

