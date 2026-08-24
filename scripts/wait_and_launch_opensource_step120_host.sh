#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RAW_ROOT="${RAW_ROOT:-/data/renjunxiang/coding/huawei_train/datasets/open_source/raw}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-trainer-m05-20260730}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-rollout-m06-20260730}"
REMOTE_HOST_PROJECT_ROOT="${REMOTE_HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-step120-opensource-20260824-01}"
DATA_SUBDIR="data/step120_opensource_20260824"
HOST_DATA_DIR="${HOST_PROJECT_ROOT}/${DATA_SUBDIR}"
CONTAINER_DATA_DIR="${CONTAINER_PROJECT_ROOT}/${DATA_SUBDIR}"
SUPERVISOR_DIR="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}-supervisor"
POLL_SECONDS="${POLL_SECONDS:-60}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
MIN_HOST_MEM_AVAILABLE_KB="${MIN_HOST_MEM_AVAILABLE_KB:-1288490189}"
MAX_HOST_MLOCKED_KB="${MAX_HOST_MLOCKED_KB:-134217728}"

mkdir -p "${SUPERVISOR_DIR}" "${HOST_DATA_DIR}"
exec 9>"${HOST_PROJECT_ROOT}/runs/.step120-opensource.lock"
if ! flock -n 9; then
  printf 'another Step-120-open-source supervisor holds the lock\n' >&2
  exit 9
fi
printf '%s\n' "$$" > "${SUPERVISOR_DIR}/supervisor.pid"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/started_at"
printf 'building_leakage_screened_curriculum\n' > "${SUPERVISOR_DIR}/state"

PYTHONPATH="${HOST_PROJECT_ROOT}:${PYTHONPATH:-}" \
python3 "${HOST_PROJECT_ROOT}/scripts/prepare_opensource_step120_data.py" build \
  --raw-root "${RAW_ROOT}" \
  --output-dir "${HOST_DATA_DIR}" \
  > "${SUPERVISOR_DIR}/data_build.log"

npu_process_pattern='^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|[[:space:]]*[[:alnum:]_]'
local_npu_busy() {
  local output
  if ! output="$(npu-smi info 2>&1)" || [[ -z "${output}" ]]; then
    return 0
  fi
  grep -Eq "${npu_process_pattern}" <<< "${output}"
}
remote_npu_busy() {
  local output
  if ! output="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" npu-smi info 2>&1)" \
    || [[ -z "${output}" ]] \
    || grep -Eq "${npu_process_pattern}" <<< "${output}"; then
    return 0
  fi
  return 1
}
local_host_memory_busy() {
  local available_kb mlocked_kb
  available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  mlocked_kb="$(awk '/^Mlocked:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  if [[ ! "${available_kb}" =~ ^[0-9]+$ || ! "${mlocked_kb}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  printf '%s\n' "${available_kb}" > "${SUPERVISOR_DIR}/last_mem_available_kb"
  printf '%s\n' "${mlocked_kb}" > "${SUPERVISOR_DIR}/last_mlocked_kb"
  (( available_kb < MIN_HOST_MEM_AVAILABLE_KB || mlocked_kb > MAX_HOST_MLOCKED_KB ))
}

printf 'waiting_for_m05_m06_idle\n' > "${SUPERVISOR_DIR}/state"
idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  if local_npu_busy || remote_npu_busy || local_host_memory_busy; then
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

printf 'starting_original_containers_and_converting_data\n' > "${SUPERVISOR_DIR}/state"
docker start "${TRAINER_CONTAINER}" > "${SUPERVISOR_DIR}/trainer_container_start.log"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker start '${ROLLOUT_CONTAINER}'" > "${SUPERVISOR_DIR}/rollout_container_start.log"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RESUME_CHECKPOINT='${CONTAINER_PROJECT_ROOT}/runs/resume-views/llin-step100-opensource/global_step_100' bash '${CONTAINER_PROJECT_ROOT}/scripts/prepare_pi_step100_resume_view.sh' trainer" \
  > "${SUPERVISOR_DIR}/trainer_resume_view.log"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"RESUME_CHECKPOINT='${CONTAINER_PROJECT_ROOT}/runs/resume-views/llin-step100-opensource/global_step_100' bash '${CONTAINER_PROJECT_ROOT}/scripts/prepare_pi_step100_resume_view.sh' rollout\"" \
  > "${SUPERVISOR_DIR}/rollout_resume_view.log"
docker exec "${TRAINER_CONTAINER}" python3 \
  "${CONTAINER_PROJECT_ROOT}/scripts/prepare_opensource_step120_data.py" convert \
  --input "${CONTAINER_DATA_DIR}/opensource_step120_train.jsonl" \
  --output "${CONTAINER_DATA_DIR}/opensource_step120_train.parquet" \
  > "${SUPERVISOR_DIR}/data_convert.log"

ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "mkdir -p '${REMOTE_HOST_PROJECT_ROOT}/${DATA_SUBDIR}'"
scp "${HOST_DATA_DIR}/opensource_step120_train.parquet" \
  "${HOST_DATA_DIR}/opensource_step120_quality_report.json" \
  "root@${ROLLOUT_HOST}:${REMOTE_HOST_PROJECT_ROOT}/${DATA_SUBDIR}/"

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
  "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_ray_roles.py' --trainer-ip 192.168.202.5 --rollout-ip 192.168.202.4" \
  > "${SUPERVISOR_DIR}/ray_roles.json"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "ROLLOUT_RANKS=16 python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_hccl_fanout.py'" \
  > "${SUPERVISOR_DIR}/hccl_fanout_1x16.json" 2>&1

printf 'training\n' > "${SUPERVISOR_DIR}/state"
set +e
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RUN_NAME='${RUN_NAME}' bash '${CONTAINER_PROJECT_ROOT}/scripts/launch_opensource_step100_to_step120.sh'" \
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
