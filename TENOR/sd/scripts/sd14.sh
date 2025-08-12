#!/usr/bin/env bash

set -Eeuo pipefail

MODEL_NAME="CompVis/stable-diffusion-v1-4"
IMGAE_RESOLUTION=512

# path to training dataset
TRAIN_DATA_DIR=set/path/to/scop/here

# set up tracker
PROJ_NAME=CoMPaSS-sd14
RUN_NAME=$PROJ_NAME-$(date +"%Y-%m-%d")

# checkpoint settings
CHECKPOINT_STEP=8000
CHECKPOINT_LIMIT=10

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=24500

# loss and lr settings
TOKEN_LOSS_SCALE=1e-3
PIXEL_LOSS_SCALE=5e-5
LEARNING_RATE=5e-6

# other settings
GRADIENT_ACCUMULATION_STEPS=2
DATALOADER_NUM_WORKERS=6

OUTPUT_DIR="results/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"

# train!
accelerate launch --main_process_port 48652 src/compass_train_unet.py \
  --pe_type="absolute" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --train_batch_size=1 \
  --resolution $IMGAE_RESOLUTION \
  --dataloader_num_workers $DATALOADER_NUM_WORKERS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing \
  --max_train_steps=$MAX_TRAINING_STEPS \
  --learning_rate $LEARNING_RATE \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR" \
  --checkpoints_total_limit $CHECKPOINT_LIMIT \
  --checkpointing_steps $CHECKPOINT_STEP \
  --token_loss_scale $TOKEN_LOSS_SCALE \
  --pixel_loss_scale $PIXEL_LOSS_SCALE \
  --train_mid 8 \
  --train_up 16 32 64 \
  --tracker_run_name "$RUN_NAME" \
  --tracker_project_name $PROJ_NAME \
