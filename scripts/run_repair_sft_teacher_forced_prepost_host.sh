#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_NAME="${CONTAINER_NAME:-llin-verl-trainer-m05-20260730}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
PIPELINE_RUN_NAME="${PIPELINE_RUN_NAME:-llin-repair-sft-teacher-forced-prepost-20260811-01}"
PIPELINE_DIR="${HOST_PROJECT_ROOT}/runs/${PIPELINE_RUN_NAME}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-llin-repair-sft-teacher-forced-step120-20260811-01}"
POST_RUN_NAME="${POST_RUN_NAME:-llin-repair-sft-teacher-forced-post-sft-20260811-01}"
BASELINE_CKPT="${BASELINE_CKPT:-${CONTAINER_PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120/actor/model/dist_ckpt}"
POST_CKPT="${POST_CKPT:-${CONTAINER_PROJECT_ROOT}/runs/llin-repair-sft-train236-overfit-step120-20260811-01/checkpoints/global_step_5/model/dist_ckpt}"
ROLLOUT_COMPARISON="${ROLLOUT_COMPARISON:-${CONTAINER_PROJECT_ROOT}/runs/llin-repair-sft-prepost-20260811-01/comparison.json}"
POLL_SECONDS="${POLL_SECONDS:-15}"

mkdir -p "${PIPELINE_DIR}"
if [[ -f "${PIPELINE_DIR}/exit_code" ]]; then
  printf 'pipeline already has exit_code: %s\n' "${PIPELINE_DIR}" >&2
  exit 2
fi
date -Iseconds > "${PIPELINE_DIR}/started_at"
FINAL_EXIT=1
finish() {
  printf '%s\n' "${FINAL_EXIT}" > "${PIPELINE_DIR}/exit_code"
  date -Iseconds > "${PIPELINE_DIR}/finished_at"
}
trap finish EXIT

set_stage() {
  printf '%s\n' "$1" > "${PIPELINE_DIR}/current_stage"
  date -Iseconds > "${PIPELINE_DIR}/stage_updated_at"
}

launch_and_wait() {
  local run_name="$1"
  local model_label="$2"
  local checkpoint="$3"
  local run_dir="${HOST_PROJECT_ROOT}/runs/${run_name}"
  if [[ ! -f "${run_dir}/exit_code" ]]; then
    docker exec \
      -e "RUN_NAME=${run_name}" \
      -e "MODEL_LABEL=${model_label}" \
      -e "SOURCE_MODEL_DIST_CKPT=${checkpoint}" \
      "${CONTAINER_NAME}" \
      bash "${CONTAINER_PROJECT_ROOT}/scripts/launch_repair_sft_teacher_forced_eval.sh"
  fi
  while [[ ! -f "${run_dir}/exit_code" ]]; do
    sleep "${POLL_SECONDS}"
  done
  local code
  code="$(tr -d '[:space:]' < "${run_dir}/exit_code")"
  if [[ "${code}" != "0" ]]; then
    printf '%s exited with %s\n' "${run_name}" "${code}" >&2
    return 1
  fi
  test -s "${run_dir}/result.json"
}

set_stage baseline_forward_only
launch_and_wait "${BASELINE_RUN_NAME}" step120 "${BASELINE_CKPT}"

set_stage post_sft_forward_only
launch_and_wait "${POST_RUN_NAME}" post_sft_step5 "${POST_CKPT}"

set_stage compare_with_free_rollout
docker exec "${CONTAINER_NAME}" python3 \
  "${CONTAINER_PROJECT_ROOT}/scripts/compare_teacher_forced_diagnostics.py" \
  --step120 "${CONTAINER_PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}/result.json" \
  --post-sft "${CONTAINER_PROJECT_ROOT}/runs/${POST_RUN_NAME}/result.json" \
  --rollout-comparison "${ROLLOUT_COMPARISON}" \
  --output "${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_RUN_NAME}/comparison.json" \
  > "${PIPELINE_DIR}/comparison_stdout.json"

set_stage done
printf 'complete\n' > "${PIPELINE_DIR}/DONE"
FINAL_EXIT=0
