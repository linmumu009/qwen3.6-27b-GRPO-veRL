#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
OUTPUT_STEM="${OUTPUT_STEM:?OUTPUT_STEM is required}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/logistics-cpt-diagnostics-20260904}"

# Direct docker exec does not inherit the interactive Ascend Python paths.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077
mkdir -p "${RUN_ROOT}/private/nll" "${RUN_ROOT}/safe/nll"

exec python3 "${PROJECT_ROOT}/scripts/run_vllm_prompt_nll.py" \
  --model "${MODEL_PATH}" \
  --model-label "${MODEL_LABEL}" \
  --cases "${RUN_ROOT}/private/nll/cases.json" \
  --tensor-parallel-size 8 \
  --max-model-len 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.8 \
  --seed 1024 \
  --private-output "${RUN_ROOT}/private/nll/${OUTPUT_STEM}.json" \
  --safe-output "${RUN_ROOT}/safe/nll/${OUTPUT_STEM}.safe.json"
