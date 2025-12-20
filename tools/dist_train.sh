#!/bin/bash
# Multi-GPU training script for ForestFormer3D

CONFIG=$1
WORK_DIR=$2
GPUS=${3:-4}  # Default to 4 GPUs
RESUME=${4:-""}  # Optional: checkpoint path to resume from

if [ -z "$CONFIG" ] || [ -z "$WORK_DIR" ]; then
    echo "Usage: $0 <config_file> <work_dir> [num_gpus] [resume_checkpoint]"
    echo "Example: $0 configs/oneformer3d_features_heidelberg.py work_dirs/heidelberg_features 4"
    echo "Example with resume: $0 configs/oneformer3d_features_heidelberg.py work_dirs/heidelberg_features 4 work_dirs/heidelberg_features/epoch_50.pth"
    exit 1
fi

cd /home/rkaharly/ForestFormer3D
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH

# Clear CUDA cache before starting
python -c "import torch; [torch.cuda.set_device(i) or torch.cuda.empty_cache() for i in range(4)]" 2>/dev/null || true

# Build command
CMD="python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR \
    --launcher pytorch"

# Add resume if provided
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Execute
eval $CMD
