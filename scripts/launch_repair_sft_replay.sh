#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
LABEL="${LABEL:?LABEL is required}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-replay-${LABEL}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
EXPECTED_EVAL_ROWS="${EXPECTED_EVAL_ROWS:-16}"
mkdir -p "${OUTPUT_DIR}"

date --iso-8601=seconds > "${OUTPUT_DIR}/started_at"
set +e
RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${PROJECT_ROOT}/scripts/run_repair_sft_replay.sh" "$@" \
  > "${OUTPUT_DIR}/driver.log" 2>&1
status=$?
set -e

validation_file="${OUTPUT_DIR}/validation/0.jsonl"
if [[ "${status}" == "0" && ! -f "${validation_file}" ]]; then
  set +e
  python3 "${PROJECT_ROOT}/scripts/copy_file_from_ray_resource.py" \
    --source "${validation_file}" \
    --output "${validation_file}" \
    --resource llin_rollout \
    --ray-address "${RAY_ADDRESS:-192.168.202.5:26379}" \
    --expected-jsonl-rows "${EXPECTED_EVAL_ROWS}" \
    >> "${OUTPUT_DIR}/driver.log" 2>&1
  recovery_status=$?
  set -e
  if [[ "${recovery_status}" != "0" ]]; then
    printf 'validation_artifact_recovery_failed\n' > "${OUTPUT_DIR}/VALIDATION_INVALID"
    status=8
  fi
fi

printf '%s\n' "${status}" > "${OUTPUT_DIR}/exit_code"
date --iso-8601=seconds > "${OUTPUT_DIR}/finished_at"
exit "${status}"
