#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-trainer-m05-20260730}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-rollout-m06-20260730}"
RUN_NAME="${RUN_NAME:-llin-grpo-candidate128-5epoch-step120-to440-20260817-01}"
SUPERVISOR_DIR="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}-supervisor"
SPLIT_DIR="${CONTAINER_PROJECT_ROOT}/runs/llin-grpo-candidate-pool-161-20260817-01/split-128-33-seed20260817"
POLL_SECONDS="${POLL_SECONDS:-60}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"

mkdir -p "${SUPERVISOR_DIR}"
exec 9>"${HOST_PROJECT_ROOT}/runs/.candidate128x5-training.lock"
if ! flock -n 9; then
  printf 'another candidate128x5 supervisor holds the lock\n' >&2
  exit 9
fi
printf '%s\n' "$$" > "${SUPERVISOR_DIR}/supervisor.pid"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/started_at"
printf 'validating_assets\n' > "${SUPERVISOR_DIR}/state"

ray_started=false
cleanup() {
  if [[ "${ray_started}" == "true" ]]; then
    set +e
    docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' \
      >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
      "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" \
      >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    date --iso-8601=seconds > "${SUPERVISOR_DIR}/ray_cleanup_finished_at"
    set -e
  fi
}
trap cleanup EXIT

validate_command="python3 '${CONTAINER_PROJECT_ROOT}/scripts/split_grpo_candidate_pool.py' validate \
  --train '${SPLIT_DIR}/train128.sensitive.parquet' \
  --test '${SPLIT_DIR}/test33.sensitive.parquet' \
  --safe-summary '${SPLIT_DIR}/split.safe.json' \
  --expected-rows 161 --expected-train-rows 128 --sandbox-root /pi_sandbox"
docker exec "${TRAINER_CONTAINER}" bash -lc "${validate_command}" \
  > "${SUPERVISOR_DIR}/trainer_asset_validation.json"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${validate_command}\"" \
  > "${SUPERVISOR_DIR}/rollout_asset_validation.json"

npu_process_pattern='^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+[[:alpha:]_]'
local_npu_busy() {
  npu-smi info | grep -Eq "${npu_process_pattern}"
}
remote_npu_busy() {
  ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" npu-smi info \
    | grep -Eq "${npu_process_pattern}"
}

printf 'waiting_for_two_hosts_idle\n' > "${SUPERVISOR_DIR}/state"
idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  if local_npu_busy || remote_npu_busy; then
    idle_checks=0
  else
    idle_checks=$((idle_checks + 1))
  fi
  printf '%s\n' "${idle_checks}" > "${SUPERVISOR_DIR}/consecutive_idle_checks"
  date --iso-8601=seconds > "${SUPERVISOR_DIR}/last_resource_check_at"
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "${POLL_SECONDS}"
  fi
done

printf 'starting_ray\n' > "${SUPERVISOR_DIR}/state"
set +e
docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' \
  > "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" \
  >> "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
set -e
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_m05.sh'" \
  > "${SUPERVISOR_DIR}/ray_m05.log" 2>&1
ray_started=true
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_m06.sh'\"" \
  > "${SUPERVISOR_DIR}/ray_m06.log" 2>&1
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_ray_roles.py'" \
  > "${SUPERVISOR_DIR}/ray_roles.json"

printf 'training\n' > "${SUPERVISOR_DIR}/state"
set +e
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RUN_NAME='${RUN_NAME}' bash '${CONTAINER_PROJECT_ROOT}/scripts/launch_pi_candidate_128x5_step120_to440.sh'" \
  > "${SUPERVISOR_DIR}/training_launcher.log" 2>&1
training_exit=$?
set -e
printf '%s\n' "${training_exit}" > "${SUPERVISOR_DIR}/exit_code"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
if [[ "${training_exit}" == "0" ]]; then
  printf 'complete\n' > "${SUPERVISOR_DIR}/state"
else
  printf 'failed\n' > "${SUPERVISOR_DIR}/state"
fi
exit "${training_exit}"
