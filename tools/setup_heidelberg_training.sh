#!/bin/bash
# Setup script for Heidelberg dataset training
# This script prepares everything needed to start training
# 
# Usage: Make sure conda environment is activated first!
#   conda activate forestformer3d
#   bash tools/setup_heidelberg_training.sh

set -e  # Exit on error

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "forestformer3d" ]; then
    echo "⚠️  Warning: Make sure to activate conda environment first:"
    echo "   conda activate forestformer3d"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=========================================="
echo "Setting up Heidelberg Dataset Training"
echo "=========================================="
echo ""

# Check if data split already exists
if [ ! -d "data/ForAINetV2_heidelberg" ]; then
    echo "Step 1: Creating data split..."
    python tools/prepare_training_splits.py --scenario heidelberg
    echo ""
else
    echo "✅ Data split already exists: data/ForAINetV2_heidelberg"
    echo ""
fi

# Copy necessary scripts
echo "Step 2: Copying data loading scripts..."
mkdir -p data/ForAINetV2_heidelberg
cp data/ForAINetV2/batch_load_ForAINetV2_data.py data/ForAINetV2_heidelberg/ 2>/dev/null || true
cp data/ForAINetV2/load_forainetv2_data.py data/ForAINetV2_heidelberg/ 2>/dev/null || true
echo "✅ Scripts copied"
echo ""

# Check if .npy files already exist
if [ ! -d "data/ForAINetV2_heidelberg/forainetv2_heidelberg_instance_data" ] || [ -z "$(ls -A data/ForAINetV2_heidelberg/forainetv2_heidelberg_instance_data/*.npy 2>/dev/null)" ]; then
    echo "Step 3: Generating .npy files from PLY files..."
    cd data/ForAINetV2_heidelberg
    # Add parent ForAINetV2 to path for imports
    PYTHONPATH=../ForAINetV2:$PYTHONPATH python batch_load_ForAINetV2_data.py \
        --train_scan_names_file meta_data/train_list.txt \
        --val_scan_names_file meta_data/val_list.txt \
        --test_scan_names_file meta_data/val_list.txt \
        --output_folder forainetv2_heidelberg_instance_data
    cd ../..
    echo "✅ .npy files generated"
    echo ""
else
    echo "✅ .npy files already exist"
    echo ""
fi

# Check if .pkl files already exist
if [ ! -f "data/ForAINetV2_heidelberg/forainetv2_heidelberg_oneformer3d_infos_train.pkl" ]; then
    echo "Step 4: Generating .pkl annotation files..."
    echo "   (This requires the conda environment to be activated)"
    export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH
    python tools/create_data_forainetv2.py forainetv2 \
        --root-path ./data/ForAINetV2_heidelberg \
        --out-dir ./data/ForAINetV2_heidelberg \
        --extra-tag forainetv2_heidelberg || {
        echo ""
        echo "⚠️  Error generating .pkl files. Make sure:"
        echo "   1. Conda environment is activated: conda activate forestformer3d"
        echo "   2. PYTHONPATH is set: export PYTHONPATH=/home/rkaharly/ForestFormer3D:\$PYTHONPATH"
        echo ""
        echo "You can run this step manually:"
        echo "   python tools/create_data_forainetv2.py forainetv2 \\"
        echo "       --root-path ./data/ForAINetV2_heidelberg \\"
        echo "       --out-dir ./data/ForAINetV2_heidelberg \\"
        echo "       --extra-tag forainetv2_heidelberg"
        exit 1
    }
    echo "✅ .pkl files generated"
    echo ""
else
    echo "✅ .pkl files already exist"
    echo ""
fi

# Verify files exist
echo "Step 5: Verifying setup..."
if [ -f "data/ForAINetV2_heidelberg/forainetv2_heidelberg_oneformer3d_infos_train.pkl" ] && \
   [ -f "data/ForAINetV2_heidelberg/forainetv2_heidelberg_oneformer3d_infos_val.pkl" ]; then
    echo "✅ All required files exist"
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete! Ready to train."
    echo "=========================================="
    echo ""
    echo "To start training, run:"
    echo ""
    echo "  export PYTHONPATH=/home/rkaharly/ForestFormer3D:\$PYTHONPATH"
    echo "  CUDA_VISIBLE_DEVICES=0 python tools/train.py \\"
    echo "      configs/oneformer3d_features_heidelberg.py \\"
    echo "      --work-dir work_dirs/heidelberg_features"
    echo ""
else
    echo "❌ Error: Some required files are missing"
    echo "   Please check the output above for errors"
    exit 1
fi

