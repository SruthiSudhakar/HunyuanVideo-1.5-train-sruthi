#!/usr/bin/env bash
set -euo pipefail

N_GPUS=8
MODEL_PATH=ckpts
CHECKPOINT_PATH=outputs/decoder_batch_32_lora_r64_jdg_largescale/checkpoint-2000
DATA_ROOT=dataset/PnPRedLegoToBrownBowl
OUTPUT_DIR=outputs/decoder_batch_32_lora_r64_jdg_largescale/generation_checkpoint-2000_pnp_red_lego_to_brown_bowl

VAL_NAME_SUBSTRINGS=(
  failure_1_20260504-215109
  # failure_2_20260504-215855
  # failure_3_20260504-220320_frames
  # failure_4_20260504-220713_frames
  # success_1_20260504-214620
  # success_2_20260504-215606
  # success_3_20260504-220109_frames
  # success_4_20260504-220533_frames
)

torchrun --standalone --nproc_per_node="${N_GPUS}" generate.py \
  --model_path "${MODEL_PATH}" \
  --transformer_version 480p_i2v \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --name_substrings "${VAL_NAME_SUBSTRINGS[@]}" \
  --max_samples 1 \
  --video_length 33 \
  --video_width 848 \
  --video_height 480 \
  --video_spatial_crop_margin 40 \
  --target_resolution 480p \
  --num_inference_steps 10 \
  --guidance_scale 1.0 \
  --seed 42 \
  --dtype bf16 \
  --sp_size "${N_GPUS}" \
  --offloading false \
  --group_offloading false \
  --overlap_group_offloading false
