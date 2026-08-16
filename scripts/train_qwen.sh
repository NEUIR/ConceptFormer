#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS="${ROOT}/configs/paths.sh"
if [[ ! -f "${PATHS}" ]]; then PATHS="${ROOT}/configs/paths.example.sh"; fi
source "${PATHS}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/ConceptFormer-Qwen}"

deepspeed --include "localhost:${GPU_IDS}" --module conceptformer.retriever.driver.train \
  --deepspeed "${ROOT}/configs/deepspeed/zero3.json" \
  --model_name_or_path "${CONCEPTFORMER_QWEN_BASE}" \
  --dataset_path "${CONCEPTFORMER_TRAIN_DATA}" \
  --corpus_path "${CONCEPTFORMER_TRAIN_CORPUS}/*.parquet" \
  --output_dir "${OUTPUT_DIR}" \
  --do_train --lora --bf16 --pooling eos --append_eos_token --normalize \
  --temperature 0.01 --query_max_len 256 --answer_max_len 256 \
  --qwen_max_pixels $((1280 * 28 * 28)) \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --num_train_epochs 3 --warmup_ratio 0.1 \
  --lora_r 8 --lora_alpha 64 --lora_dropout 0.1 \
  --kl_loss_weight 0 \
  --latent_align_mode forward --latent_lambda_forward 0.2 \
  --latent_lambda_reverse 0 --latent_mse_weight 0 \
  --latent_kl_variant concept2image --latent_pooling mean --latent_num_tokens 0 \
  --logging_steps 10 --save_strategy epoch --save_total_limit 1 \
  --dataloader_num_workers 8 --remove_unused_columns false --report_to none \
  --seed 42
