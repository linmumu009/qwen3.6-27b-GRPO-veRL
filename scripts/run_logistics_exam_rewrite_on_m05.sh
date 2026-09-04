#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource}"
MODEL_LABEL="${MODEL_LABEL:-qwen3.6-27b-step120-local-rewriter}"
SOURCE_PATH="${SOURCE_PATH:-${PROJECT_ROOT}/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl}"
PRIVATE_OUTPUT="${PRIVATE_OUTPUT:-${PROJECT_ROOT}/runs/logistics-exam-cpt-20260904/private/rewritten-stems.jsonl}"
SAFE_OUTPUT="${SAFE_OUTPUT:-${PROJECT_ROOT}/runs/logistics-exam-cpt-20260904/safe/rewritten-stems.safe.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077

exec python3 "${PROJECT_ROOT}/scripts/rewrite_logistics_exam_stems_offline.py" \
  --source "${SOURCE_PATH}" \
  --model "${MODEL_PATH}" \
  --model-label "${MODEL_LABEL}" \
  --tensor-parallel-size 8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization 0.8 \
  --seed 20260904 \
  --max-attempts 4 \
  --private-output "${PRIVATE_OUTPUT}" \
  --safe-output "${SAFE_OUTPUT}" \
  "$@"
