#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <huggingface-organization>"
  exit 2
fi
ORG="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../huggingface" && pwd)"
for NAME in ConceptFormer-Trainset ConceptFormer-Eval; do
  huggingface-cli upload-large-folder "${ORG}/${NAME}" "${ROOT}/${NAME}" --repo-type dataset
done
for NAME in ConceptFormer-Qwen ConceptFormer-Phi3V; do
  huggingface-cli upload-large-folder "${ORG}/${NAME}" "${ROOT}/${NAME}"
done
