#!/usr/bin/env bash
set -euo pipefail
ORG="${1:-hmhm1229}"
ROOT="${2:-data}"
mkdir -p "${ROOT}"
huggingface-cli download "${ORG}/ConceptFormer-Trainset" --repo-type dataset \
  --local-dir "${ROOT}/train"
huggingface-cli download "${ORG}/ConceptFormer-Eval" --repo-type dataset \
  --local-dir "${ROOT}/eval"
echo "ConceptFormer-Eval includes image-complete corpus Parquet shards for all six benchmarks."
