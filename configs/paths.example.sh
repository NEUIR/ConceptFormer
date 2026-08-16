#!/usr/bin/env bash

# Copy this file to configs/paths.sh and adjust only the paths you need.
export CONCEPTFORMER_QWEN_BASE="${CONCEPTFORMER_QWEN_BASE:-Qwen/Qwen2.5-VL-7B-Instruct}"
export CONCEPTFORMER_PHI3V_BASE="${CONCEPTFORMER_PHI3V_BASE:-Tevatron/dse-phi3-docmatix-v1}"
export CONCEPTFORMER_PHI3V_INIT_ADAPTER="${CONCEPTFORMER_PHI3V_INIT_ADAPTER:-NTT-hil-insight/VDocRetriever-Phi3-vision-pretrained}"
export CONCEPTFORMER_TRAIN_DATA="${CONCEPTFORMER_TRAIN_DATA:-data/train/data/train.parquet}"
export CONCEPTFORMER_TRAIN_CORPUS="${CONCEPTFORMER_TRAIN_CORPUS:-data/train_corpus}"
export CONCEPTFORMER_EVAL_DATA="${CONCEPTFORMER_EVAL_DATA:-data/eval/data}"
export CONCEPTFORMER_OPENDOC_CORPUS="${CONCEPTFORMER_OPENDOC_CORPUS:-data/opendoc_corpus}"
export CONCEPTFORMER_QWEN_ADAPTER="${CONCEPTFORMER_QWEN_ADAPTER:-hmhm1229/ConceptFormer-Qwen}"
export CONCEPTFORMER_PHI3V_ADAPTER="${CONCEPTFORMER_PHI3V_ADAPTER:-hmhm1229/ConceptFormer-Phi3V}"
