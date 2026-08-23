#!/bin/bash

# Stage 1: Foundational Pretraining
# Text Corpus 1 (62M multilingual Wikipedia spans) + 26M LAION image-text pairs.

RUN="pretrain"

BASE_MODEL="./Qwen2.5-VL-ViT-Only"
DATA_PATH="./data/laion-image-text"
OUTPUT_DIR="checkpoints/${RUN}"

BATCH_SIZE=1024
PER_DEVICE_BATCH_SIZE=512
EPOCH=2
LR=5e-5
MAX_LENGTH=650

NNODES=1
NPROC_PER_NODE=2

echo "Starting run: ${RUN}"
echo "Base Model: ${BASE_MODEL}"
echo "Data Path: ${DATA_PATH}"
echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}, Total Batch size: ${BATCH_SIZE}"

export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1

# For multi-node, add: --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    train_laion_inbatch_multilingual.py \
    --unsupervised_data_path './data/wikispan-filtered' \
    --base_model $BASE_MODEL \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $BATCH_SIZE \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --max_length ${MAX_LENGTH} \
    --save_steps 200 \
    --logging_steps 1 \
    --bf16 \
    --deepspeed deepspeed_config/ds.config
