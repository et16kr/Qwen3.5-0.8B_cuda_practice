#!/bin/sh
set -eu

MODEL_DIR="${MODEL_DIR:-../Qwen3.5-0.8B}"
INPUT_TXT="${INPUT_TXT:-./data/requests.txt}"
OUTPUT_TXT="${OUTPUT_TXT:-./data/responses.txt}"
MAIN_BINARY="${MAIN_BINARY:-./main}"

python3 ./scripts/run_text_generation.py \
  --model-dir "${MODEL_DIR}" \
  --input "${INPUT_TXT}" \
  --output "${OUTPUT_TXT}" \
  --main-binary "${MAIN_BINARY}"
