#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
DATA=${ROOT}/runs/logistics-reviewed8-20260905
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
exec 8>"${DATA}/pipeline.lock"
flock -n 8
BASE=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
python3 "${ROOT}/scripts/check_logistics_mask8.py" --data "${DATA}/same-text.parquet" \
  --model "${BASE}" --output "${DATA}/mask-gate.safe.json"
python3 "${ROOT}/scripts/probe_logistics_exam_answers.py" --model "${BASE}" --label step120 \
  --items "${DATA}/items.jsonl" --output "${DATA}/step120.json" > "${DATA}/step120.eval.log" 2>&1
export TRAIN_FILE=${DATA}/same-text.parquet
export EXPECTED_TRAIN_FILE_SHA256=$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')
export EXPECTED_CONTENT_TOKENS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_tokens"])' "${DATA}/manifest.safe.json")
for ARM in all answer; do
  export ARM
  bash "${ROOT}/scripts/run_logistics_mask8_arm.sh" > "${DATA}/${ARM}.train.log" 2>&1
  python3 "${ROOT}/scripts/probe_logistics_exam_answers.py" \
    --model "${ROOT}/runs/logistics-mask8-${ARM}-20260905/hf_export_step_64" --label "${ARM}" \
    --items "${DATA}/items.jsonl" --output "${DATA}/${ARM}.json" > "${DATA}/${ARM}.eval.log" 2>&1
done
printf 'complete\n' > "${DATA}/pipeline-status.txt"
