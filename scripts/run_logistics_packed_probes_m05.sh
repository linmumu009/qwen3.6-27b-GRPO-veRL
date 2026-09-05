#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
umask 077
OUT=${ROOT}/runs/logistics-answer-probes-20260905
exec 9>"${OUT}/knowledge.lock"
flock 9
for label in step120 direct rewritten; do
  if [[ ${label} == step120 ]]; then
    model=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
  else
    model=${ROOT}/runs/logistics-exam-cpt-${label}-4x-20260904-01/hf_export_step_64
  fi
  if [[ -f ${OUT}/${label}.packed.safe.json ]]; then continue; fi
  python3 "${ROOT}/scripts/probe_logistics_packed_answers.py" --model "${model}" --label "${label}" \
    --corpus-root "${ROOT}/runs/logistics-exam-cpt-20260904/private" \
    --output "${OUT}/${label}.packed.safe.json" > "${OUT}/${label}.packed.log" 2>&1
done
printf 'complete\n' > "${OUT}/packed-status.txt"
