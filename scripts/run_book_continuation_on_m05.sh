#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
OUTPUT_STEM="${OUTPUT_STEM:?OUTPUT_STEM is required}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/logistics-cpt-diagnostics-20260904}"
CASE_ROOT="${CASE_ROOT:-${RUN_ROOT}/private/continuation/cases}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077

exec python3 "${PROJECT_ROOT}/scripts/run_vllm_book_continuation.py" \
  --model "${MODEL_PATH}" \
  --model-label "${MODEL_LABEL}" \
  --cases "32=${CASE_ROOT}/p32_n200.jsonl" \
  --cases "64=${CASE_ROOT}/p64_n200.jsonl" \
  --cases "128=${CASE_ROOT}/p128_n200.jsonl" \
  --output-dir "${RUN_ROOT}/private/continuation/${OUTPUT_STEM}" \
  --safe-dir "${RUN_ROOT}/safe/continuation/${OUTPUT_STEM}" \
  --tensor-parallel-size 8 \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --max-output-tokens 96 \
  --gpu-memory-utilization 0.8 \
  --seed 1024
