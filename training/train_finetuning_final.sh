#!/bin/bash

# Stage 3 (optional): AllNLI triplet finetuning
# ~270K AllNLI triplets, rendered to images in the collator.

RUN="midtrain-ft"

BASE_MODEL="./checkpoints/midtrain"
DATA_PATH="./data/all_nli_triplets.json"
OUTPUT_DIR="checkpoints-finetuning/${RUN}"

BATCH_SIZE=768
PER_DEVICE_BATCH_SIZE=96
EPOCH=2
LR=5e-6
MAX_LENGTH=650

NNODES=2
NPROC_PER_NODE=4

echo "Starting run: ${RUN}"
echo "Base Model: ${BASE_MODEL}"
echo "Data Path: ${DATA_PATH}"
echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}, Total Batch size: ${BATCH_SIZE}"

export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_finetuning.py \
    --base_model $BASE_MODEL \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $BATCH_SIZE \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --max_length ${MAX_LENGTH} \
    --save_steps 100 \
    --logging_steps 1 \
    --bf16 \
    --deepspeed deepspeed_config/ds.config
