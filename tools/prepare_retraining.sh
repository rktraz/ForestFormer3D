#!/bin/bash
# Prepare for retraining from scratch

WORK_DIR="work_dirs/heidelberg_features"

echo "🧹 Cleaning up old training artifacts..."

# Delete old checkpoints (incompatible weight format)
if [ -d "$WORK_DIR" ]; then
    echo "  Deleting old checkpoints..."
    rm -f "$WORK_DIR"/*.pth 2>/dev/null
    echo "  ✅ Deleted checkpoints"
    
    # Keep logs for reference (optional - uncomment to delete)
    # rm -f "$WORK_DIR"/*.log 2>/dev/null
    
    echo "  Keeping work directory structure"
else
    echo "  Work directory doesn't exist, will be created during training"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 What was kept:"
echo "  - Training data: data/ForAINetV2_heidelberg/ (6.0GB)"
echo "  - Config files: configs/oneformer3d_features_heidelberg.py"
echo "  - Data splits: train_list.txt, val_list.txt"
echo "  - Annotation files: *.pkl"
echo ""
echo "🚀 Ready to retrain! Run:"
echo "  bash tools/dist_train.sh configs/oneformer3d_features_heidelberg.py work_dirs/heidelberg_features 4"
