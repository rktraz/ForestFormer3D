#!/usr/bin/env bash
# Inference script for a single test file with explicit path specification
# Usage: ./tools/run_inference_single_file.sh <path_to_input_file> [output_dir]

set -euo pipefail

# Get input file path
INPUT_FILE="${1:-}"
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file specified"
    echo "Usage: $0 <path_to_input_file> [output_dir]"
    echo "Example: $0 test_files/256_leafon.las"
    echo "Example: $0 data/my_file.ply"
    exit 1
fi

# Try to resolve path (handle relative paths)
if [ ! -f "$INPUT_FILE" ]; then
    # Try relative to current directory
    if [ -f "$(pwd)/$INPUT_FILE" ]; then
        INPUT_FILE="$(pwd)/$INPUT_FILE"
    # Try relative to script directory
    elif [ -f "$SCRIPT_DIR/../$INPUT_FILE" ]; then
        INPUT_FILE="$(cd "$SCRIPT_DIR/.." && pwd)/$INPUT_FILE"
    else
        echo "Error: Input file not found: $INPUT_FILE"
        echo ""
        echo "Searched locations:"
        echo "  - $INPUT_FILE"
        echo "  - $(pwd)/$INPUT_FILE"
        echo "  - $(cd "$SCRIPT_DIR/.." && pwd)/$INPUT_FILE"
        echo ""
        echo "Please provide the full path to your file, for example:"
        echo "  $0 /full/path/to/256_leafon.las"
        echo "  $0 ~/path/to/256_leafon.las"
        echo "  $0 ./test_files/256_leafon.las"
        exit 1
    fi
fi

# Get absolute path
INPUT_FILE=$(realpath "$INPUT_FILE" 2>/dev/null || echo "$INPUT_FILE")
echo "Using input file: $INPUT_FILE"

# Get working directory (use absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$WORK_DIR"
echo "Working directory: $WORK_DIR"
echo "Current directory: $(pwd)"

# Output directory (optional second argument)
OUTPUT_DIR="${2:-work_dirs/output}"

# Test data directory
TEST_DATA_DIR="$WORK_DIR/data/ForAINetV2/test_data"
META_DIR="$WORK_DIR/data/ForAINetV2/meta_data"
CONFIG_FILE="$WORK_DIR/configs/oneformer3d_qs_radius16_qp300_2many.py"
MODEL_PATH="$WORK_DIR/work_dirs/clean_forestformer/epoch_3000_fix.pth"

# Create directories
mkdir -p "$TEST_DATA_DIR" "$META_DIR" "$OUTPUT_DIR"

# Get file extension
INPUT_EXT="${INPUT_FILE##*.}"
INPUT_EXT_LOWER="${INPUT_EXT,,}"

# Get base filename without extension
BASE_NAME=$(basename "$INPUT_FILE" ".$INPUT_EXT")

# Convert LAS/LAZ to PLY if needed
if [[ "$INPUT_EXT_LOWER" == "las" || "$INPUT_EXT_LOWER" == "laz" ]]; then
    echo "Converting LAS/LAZ to PLY format..."
    PLY_FILE="$TEST_DATA_DIR/${BASE_NAME}.ply"
    
    # Activate conda to use laspy (disable unbound variable check for conda)
    set +u  # Temporarily disable unbound variable checking
    source /home/rkaharly/miniconda/etc/profile.d/conda.sh
    conda activate forestformer3d
    set -u  # Re-enable unbound variable checking
    
    python tools/convert_las_to_ply.py "$INPUT_FILE" "$PLY_FILE"
    
    if [ ! -f "$PLY_FILE" ]; then
        echo "Error: Conversion failed"
        exit 1
    fi
    echo "✅ Converted to: $PLY_FILE"
else
    # Copy PLY file to test_data directory
    PLY_FILE="$TEST_DATA_DIR/${BASE_NAME}.ply"
    echo "Copying file to test_data directory..."
    cp "$INPUT_FILE" "$PLY_FILE"
    echo "✅ Copied to: $PLY_FILE"
fi

# Clean temporary data
echo "[Step 0] Cleaning temporary data files..."
rm -rf "$WORK_DIR/data/ForAINetV2/forainetv2_instance_data"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/semantic_mask"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/points"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/instance_mask"/* || true

# Generate test_list.txt with only this file
echo "[Step 1] Generating test_list.txt..."
TEST_LIST="$META_DIR/test_list.txt"
TEST_LIST_INIT="$META_DIR/test_list_initial.txt"

echo "$BASE_NAME" > "$TEST_LIST"
cp -f "$TEST_LIST" "$TEST_LIST_INIT"
echo "✅ Created test_list.txt with: $BASE_NAME"

# Activate conda environment (fix unbound variable issue)
set +u  # Temporarily disable unbound variable checking
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
set -u  # Re-enable unbound variable checking

# Set environment variables
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ITERATIONS="${ITERATIONS:-1}"

# Run data preprocessing
echo "[Step 2] Running data preprocessing..."
cd "$WORK_DIR/data/ForAINetV2"
python batch_load_ForAINetV2_data.py --test_scan_names_file meta_data/test_list.txt
cd "$WORK_DIR"

# Create data info files
echo "[Step 3] Creating data info files..."
python "$WORK_DIR/tools/create_data_forainetv2.py" forainetv2

# Update data info files
echo "[Step 4] Updating data info files..."
# Note: update_infos_to_v2.py is already called by create_data_forainetv2.py, so this step is not needed
# python "$WORK_DIR/tools/update_infos_to_v2.py" --dataset forainetv2 --pkl-path "$META_DIR/forainetv2_oneformer3d_infos_test.pkl" --root-dir "$WORK_DIR/data/ForAINetV2" --out-dir "$WORK_DIR/data/ForAINetV2"

# Run inference using the existing pipeline
echo "[Step 5] Running inference..."
export CURRENT_ITERATION=1

# Run model inference
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$WORK_DIR/tools/test.py" "$CONFIG_FILE" "$MODEL_PATH"

# Check for output and merge if needed
if [ -f "$OUTPUT_DIR/${BASE_NAME}_1.ply" ]; then
    echo "[Step 6] Merging predictions..."
    python tools/merge_prediction.py "$BASE_NAME" "$OUTPUT_DIR" "$ITERATIONS"
else
    echo "Warning: Output file not found at $OUTPUT_DIR/${BASE_NAME}_1.ply"
    echo "Checking for alternative output locations..."
    find "$OUTPUT_DIR" -name "*${BASE_NAME}*" -type f 2>/dev/null | head -5
fi

echo ""
echo "✅ Inference complete!"
echo "Results are in: $OUTPUT_DIR/round_1/"
echo "Main output: $OUTPUT_DIR/round_1/${BASE_NAME}_round1.ply"
echo "Filtered output: $OUTPUT_DIR/round_1_after_remove_noise_200/${BASE_NAME}_round1.ply"

