#!/bin/bash
# Multi-GPU training script that waits for GPUs to become available

CONFIG=$1
WORK_DIR=$2
RESUME=${3:-""}
MAX_WAIT=${4:-300}  # Max wait time in seconds (default 5 minutes)

if [ -z "$CONFIG" ] || [ -z "$WORK_DIR" ]; then
    echo "Usage: $0 <config_file> <work_dir> [resume_checkpoint] [max_wait_seconds]"
    exit 1
fi

cd /home/rkaharly/ForestFormer3D
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
export PYTHONPATH=/home/rkaharly/ForestFormer3D:$PYTHONPATH

echo "Checking GPU availability for 4-GPU training..."
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    AVAILABLE=0
    for i in 0 1 2 3; do
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$i)
        if [ "$FREE_MEM" -gt 10000 ]; then  # At least 10GB free
            AVAILABLE=$((AVAILABLE + 1))
        fi
    done
    
    if [ $AVAILABLE -eq 4 ]; then
        echo "✅ All 4 GPUs are available! Starting training..."
        break
    else
        echo "⏳ Waiting for GPUs... ($AVAILABLE/4 available, waited ${WAIT_TIME}s)"
        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
    fi
done

if [ $AVAILABLE -lt 4 ]; then
    echo "❌ Timeout: Only $AVAILABLE/4 GPUs available after ${MAX_WAIT}s"
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader
    exit 1
fi

# Clear CUDA cache
python -c "import torch; [torch.cuda.set_device(i) or torch.cuda.empty_cache() for i in range(4)]" 2>/dev/null || true

# Build command
CMD="python -m torch.distributed.launch \
    --nproc_per_node=4 \
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
echo "Starting 4-GPU training..."
eval $CMD
