#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-logistics-cpt-book-exposure-curve-2x4x-20260904-01}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_NAME}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:/verl:${PYTHONPATH:-}"
export TRANSFORMERS_VERBOSITY=error
umask 077

bash "${PROJECT_ROOT}/scripts/run_logistics_cpt_book_exposure_curve.sh"

python3 "${PROJECT_ROOT}/scripts/summarize_logistics_cpt_exposure_curve.py" \
  --run-dir "${RUN_ROOT}" \
  --checkpoint-exposure 2 \
  --checkpoint-exposure 4 \
  --output "${RUN_ROOT}/training_summary.safe.json"

for step in 58 116; do
  python3 "${PROJECT_ROOT}/scripts/export_megatron_dist_to_hf.py" \
    --actor-checkpoint "${RUN_ROOT}/checkpoints/global_step_${step}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${RUN_ROOT}/hf_export_step_${step}"
done

printf 'complete\n' > "${RUN_ROOT}/pipeline_status.txt"
