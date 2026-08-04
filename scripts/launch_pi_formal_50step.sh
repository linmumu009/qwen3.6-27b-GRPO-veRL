#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-pi-formal-grpo-4of4-50step-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "$$" > "${RUN_DIR}/driver.pid"
date --iso-8601=seconds > "${RUN_DIR}/started_at"

set +e
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${RUN_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_pi_formal_50step.sh" \
  > "${RUN_DIR}/driver.log" 2>&1
exit_code=$?
set -e

# veRL/mbridge can return success even when a PP checkpoint omits a pipeline
# stage.  Validate the last checkpoint before publishing a successful exit
# code so supervisors cannot promote a run with unusable weights.
latest_iteration_file="${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt"
if [[ "${exit_code}" == "0" && -f "${latest_iteration_file}" ]]; then
  latest_iteration="$(tr -d '[:space:]' < "${latest_iteration_file}")"
  checkpoint_dir="${RUN_DIR}/checkpoints/global_step_${latest_iteration}"
  set +e
  python3 "${PROJECT_ROOT}/scripts/verify_checkpoint_integrity.py" \
    --checkpoint-dir "${checkpoint_dir}" \
    --base-model-dir "${BASE_MODEL_DIR:-/models/Qwen3.6-27B}" \
    --output "${RUN_DIR}/checkpoint_integrity.json" \
    >> "${RUN_DIR}/driver.log" 2>&1
  checkpoint_exit=$?
  set -e
  if [[ "${checkpoint_exit}" != "0" ]]; then
    printf 'checkpoint_integrity_failed\n' > "${RUN_DIR}/CHECKPOINT_INVALID"
    exit_code=8
  fi
fi

printf '%s\n' "${exit_code}" > "${RUN_DIR}/exit_code"
date --iso-8601=seconds > "${RUN_DIR}/finished_at"
exit "${exit_code}"
