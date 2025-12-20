#!/bin/bash
# Auto-detect available GPUs and use them for training

CONFIG=$1
WORK_DIR=$2
RESUME=${3:-""}  # Optional: checkpoint path to resume from

if [ -z "$CONFIG" ] || [ -z "$WORK_DIR" ]; then
    echo "Usage: $0 <config_file> <work_dir> [resume_checkpoint]"
    exit 1
fi

cd /home/rkaharly/ForestFormer3D
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH

# Check GPU availability
AVAILABLE_GPUS=()
for i in 0 1 2 3; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$i)
    if [ "$FREE_MEM" -gt 8000 ]; then  # At least 8GB free
        AVAILABLE_GPUS+=($i)
    fi
done

NUM_GPUS=${#AVAILABLE_GPUS[@]}

if [ $NUM_GPUS -eq 0 ]; then
    echo "❌ No GPUs with sufficient free memory (need >8GB)"
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
    exit 1
fi

echo "✅ Found $NUM_GPUS available GPU(s): ${AVAILABLE_GPUS[@]}"

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Using single GPU: ${AVAILABLE_GPUS[0]}"
    GPU_LIST="${AVAILABLE_GPUS[0]}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST python tools/train.py $CONFIG --work-dir $WORK_DIR"
else
    # Multi-GPU training
    GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
    echo "Using GPUs: $GPU_LIST"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tools/train.py \
        $CONFIG \
        --work-dir $WORK_DIR \
        --launcher pytorch"
fi

# Add resume if provided
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Execute
eval $CMD
