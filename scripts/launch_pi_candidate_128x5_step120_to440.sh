#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
FINAL_POLICY_STEP=440
EXPECTED_TEST_ROWS=33
RUN_NAME="${RUN_NAME:-llin-grpo-candidate128-5epoch-step120-to440-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "$$" > "${RUN_DIR}/driver.pid"
date --iso-8601=seconds > "${RUN_DIR}/started_at"

set +e
RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${RUN_DIR}" \
  bash "${PROJECT_ROOT}/scripts/run_pi_candidate_128x5_step120_to440.sh" \
  > "${RUN_DIR}/driver.log" 2>&1
exit_code=$?
set -e

validation_file="${RUN_DIR}/validation/${FINAL_POLICY_STEP}.jsonl"
if [[ "${exit_code}" == "0" && ! -f "${validation_file}" ]]; then
  set +e
  python3 "${PROJECT_ROOT}/scripts/copy_file_from_ray_resource.py" \
    --source "${validation_file}" \
    --output "${validation_file}" \
    --resource llin_rollout \
    --ray-address "${RAY_ADDRESS:-192.168.202.5:26379}" \
    --expected-jsonl-rows "${EXPECTED_TEST_ROWS}" \
    >> "${RUN_DIR}/driver.log" 2>&1
  validation_exit=$?
  set -e
  if [[ "${validation_exit}" != "0" ]]; then
    printf 'validation_artifact_recovery_failed\n' > "${RUN_DIR}/VALIDATION_INVALID"
    exit_code=8
  fi
fi

latest_iteration_file="${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt"
if [[ "${exit_code}" == "0" ]]; then
  if [[ ! -f "${latest_iteration_file}" ]]; then
    printf 'final_checkpoint_missing\n' > "${RUN_DIR}/CHECKPOINT_INVALID"
    exit_code=8
  else
    latest_iteration="$(tr -d '[:space:]' < "${latest_iteration_file}")"
    if [[ "${latest_iteration}" != "${FINAL_POLICY_STEP}" ]]; then
      printf 'expected_global_step_%s_got_%s\n' \
        "${FINAL_POLICY_STEP}" "${latest_iteration}" > "${RUN_DIR}/CHECKPOINT_INVALID"
      exit_code=8
    fi
  fi
fi

if [[ "${exit_code}" == "0" ]]; then
  set +e
  python3 "${PROJECT_ROOT}/scripts/verify_checkpoint_integrity.py" \
    --checkpoint-dir "${RUN_DIR}/checkpoints/global_step_${FINAL_POLICY_STEP}" \
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
