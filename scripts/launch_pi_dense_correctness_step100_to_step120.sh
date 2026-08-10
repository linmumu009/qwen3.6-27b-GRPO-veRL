#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-pi-dense-correctness-step100-to-step120-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "$$" > "${RUN_DIR}/driver.pid"
date --iso-8601=seconds > "${RUN_DIR}/started_at"

set +e
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${RUN_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_pi_dense_correctness_step100_to_step120.sh" \
  > "${RUN_DIR}/driver.log" 2>&1
exit_code=$?
set -e

latest_iteration_file="${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt"
if [[ "${exit_code}" == "0" ]]; then
  if [[ ! -f "${latest_iteration_file}" ]]; then
    printf 'final_checkpoint_missing\n' > "${RUN_DIR}/CHECKPOINT_INVALID"
    exit_code=8
  else
    latest_iteration="$(tr -d '[:space:]' < "${latest_iteration_file}")"
    if [[ "${latest_iteration}" != "120" ]]; then
      printf 'expected_global_step_120_got_%s\n' "${latest_iteration}" \
        > "${RUN_DIR}/CHECKPOINT_INVALID"
      exit_code=8
    else
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
  fi
fi

printf '%s\n' "${exit_code}" > "${RUN_DIR}/exit_code"
date --iso-8601=seconds > "${RUN_DIR}/finished_at"
exit "${exit_code}"
