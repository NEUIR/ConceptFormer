#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <adapter> <base-model> <gpu-ids> [datasets]"
  exit 2
fi

ADAPTER="$1"
BASE_MODEL="$2"
IFS=',' read -r -a GPUS <<< "$3"
DATASETS="${4:-infovqa,chartqa,slidevqa,tqa,owid_charts_en,wikimedia-commons-maps}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS="${ROOT}/configs/paths.sh"
if [[ ! -f "${PATHS}" ]]; then PATHS="${ROOT}/configs/paths.example.sh"; fi
source "${PATHS}"

EVAL_ROOT="${CONCEPTFORMER_EVAL_DATA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/$(basename "${ADAPTER}")}"
BATCH_SIZE="${BATCH_SIZE:-2}"
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a NAMES <<< "${DATASETS}"
for DATASET in "${NAMES[@]}"; do
  QUERY="${EVAL_ROOT}/${DATASET}/query.parquet"
  CORPUS="${EVAL_ROOT}/${DATASET}/corpus.parquet"
  CORPUS_PATTERN="${EVAL_ROOT}/${DATASET}/corpus*.parquet"
  QRELS="${EVAL_ROOT}/${DATASET}/qrels.txt"
  CORPUS_ARGS=(--corpus_name parquet --corpus_path "${CORPUS}" --corpus_split train)
  if compgen -G "${CORPUS_PATTERN}" >/dev/null; then
    CORPUS_ARGS=(--corpus_name parquet --corpus_path "${CORPUS_PATTERN}" --corpus_split train)
  elif [[ ! -f "${CORPUS}" && "${DATASET}" =~ ^(infovqa|chartqa|slidevqa)$ ]]; then
    LOCAL_PATTERN="${CONCEPTFORMER_OPENDOC_CORPUS}/${DATASET}/*.parquet"
    if compgen -G "${LOCAL_PATTERN}" >/dev/null; then
      CORPUS_ARGS=(--corpus_name parquet --corpus_path "${LOCAL_PATTERN}" --corpus_split train)
    else
      CORPUS_ARGS=(--corpus_name NTT-hil-insight/OpenDocVQA-Corpus --corpus_config "${DATASET}" --corpus_split test)
    fi
  elif [[ ! -f "${CORPUS}" ]]; then
    echo "Missing corpus Parquet for ${DATASET}. Download ConceptFormer-Eval or run scripts/package_eval_corpora.py."
    exit 1
  fi
  [[ -f "${QUERY}" ]] || { echo "Missing query file: ${QUERY}"; exit 1; }
  [[ -f "${QRELS}" ]] || { echo "Missing qrels file: ${QRELS}"; exit 1; }

  DS_OUT="${OUTPUT_ROOT}/${DATASET}"
  mkdir -p "${DS_OUT}"
  PIDS=()
  for I in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$I]}" python -m conceptformer.retriever.driver.encode \
      --output_dir "${DS_OUT}/tmp" --model_name_or_path "${BASE_MODEL}" \
      --lora_name_or_path "${ADAPTER}" --lora --bf16 --pooling eos \
      --append_eos_token --normalize --per_device_eval_batch_size "${BATCH_SIZE}" \
      "${CORPUS_ARGS[@]}" \
      --dataset_number_of_shards "${#GPUS[@]}" --dataset_shard_index "${I}" \
      --encode_output_path "${DS_OUT}/corpus.${I}.pkl" \
      > "${DS_OUT}/corpus.${I}.log" 2>&1 &
    PIDS+=("$!")
  done
  for PID in "${PIDS[@]}"; do wait "${PID}"; done

  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m conceptformer.retriever.driver.encode \
    --output_dir "${DS_OUT}/tmp" --model_name_or_path "${BASE_MODEL}" \
    --lora_name_or_path "${ADAPTER}" --lora --bf16 --pooling eos \
    --append_eos_token --normalize --encode_is_query \
    --per_device_eval_batch_size "${BATCH_SIZE}" --query_max_len 256 \
    --dataset_name parquet --dataset_path "${QUERY}" --dataset_split train \
    --encode_output_path "${DS_OUT}/query.pkl"

  python -m conceptformer.retriever.driver.search \
    --query_reps "${DS_OUT}/query.pkl" --document_reps "${DS_OUT}/corpus.*.pkl" \
    --depth 1000 --batch_size 64 --save_text --save_ranking_to "${DS_OUT}/ranking.txt"

  python "${ROOT}/scripts/eval_metrics.py" \
    --qrels "${QRELS}" --ranking "${DS_OUT}/ranking.txt" \
    --model "$(basename "${ADAPTER}")" --dataset "${DATASET}" \
    --output "${OUTPUT_ROOT}/metrics.csv"
done
