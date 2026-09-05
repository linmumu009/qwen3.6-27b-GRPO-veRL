#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
umask 077
OUT=${ROOT}/runs/logistics-answer-probes-20260905
mkdir -p "${OUT}"
exec 9>"${OUT}/pipeline.lock"
flock 9
for label in step120 direct rewritten; do
  if [[ ${label} == step120 ]]; then
    model=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
  else
    model=${ROOT}/runs/logistics-exam-cpt-${label}-4x-20260904-01/hf_export_step_64
  fi
  if [[ -f ${OUT}/${label}.safe.json ]]; then continue; fi
  python3 "${ROOT}/scripts/probe_logistics_exam_answers.py" --model "${model}" --label "${label}" \
    --items "${ROOT}/runs/logistics-exam-cpt-20260904/private/direct-balanced.items.jsonl" \
    --items "${ROOT}/runs/logistics-exam-cpt-20260904/private/rewritten-balanced.items.jsonl" \
    --output "${OUT}/${label}.json" > "${OUT}/${label}.log" 2>&1
done
printf 'complete\n' > "${OUT}/status.txt"
