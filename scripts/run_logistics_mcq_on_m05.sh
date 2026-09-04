#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
CASES_PATH="${CASES_PATH:?CASES_PATH is required}"
PRIVATE_OUTPUT="${PRIVATE_OUTPUT:?PRIVATE_OUTPUT is required}"
SAFE_OUTPUT="${SAFE_OUTPUT:?SAFE_OUTPUT is required}"
REPEATS="${REPEATS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-96}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077

mkdir -p "$(dirname "${PRIVATE_OUTPUT}")" "$(dirname "${SAFE_OUTPUT}")"

exec python3 "${PROJECT_ROOT}/scripts/run_vllm_logistics_mcq.py" \
  --model "${MODEL_PATH}" \
  --model-label "${MODEL_LABEL}" \
  --cases "${CASES_PATH}" \
  --tensor-parallel-size 8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 64 \
  --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  --gpu-memory-utilization 0.8 \
  --seed 1024 \
  --repeats "${REPEATS}" \
  --private-output "${PRIVATE_OUTPUT}" \
  --safe-output "${SAFE_OUTPUT}"
