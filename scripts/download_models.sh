#!/usr/bin/env bash
set -euo pipefail
ORG="${1:-hmhm1229}"
ROOT="${2:-models}"
mkdir -p "${ROOT}"
huggingface-cli download "${ORG}/ConceptFormer-Qwen" --local-dir "${ROOT}/ConceptFormer-Qwen"
huggingface-cli download "${ORG}/ConceptFormer-Phi3V" --local-dir "${ROOT}/ConceptFormer-Phi3V"
