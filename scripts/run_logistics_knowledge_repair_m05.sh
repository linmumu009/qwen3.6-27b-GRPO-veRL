#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
umask 077
OUT=${ROOT}/runs/logistics-answer-probes-20260905
mkdir -p "${OUT}"
exec 9>"${OUT}/knowledge.lock"
flock 9
python3 "${ROOT}/scripts/repair_logistics_exam_knowledge.py" \
  --source "${ROOT}/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl" \
  --model "${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource" \
  --output "${OUT}/knowledge-candidates.jsonl" > "${OUT}/knowledge-repair.log" 2>&1
printf 'complete\n' > "${OUT}/knowledge-status.txt"
