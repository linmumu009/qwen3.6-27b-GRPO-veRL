#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-qwen38-smoke-m05-20260817}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-qwen38-smoke-m06-20260817}"
RUN_NAME="${RUN_NAME:-llin-v15-mixed21-4x-strict-kl-step120-20260822-01}"
RUN_HOST="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}"
RUN_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/${RUN_NAME}"
SUPERVISOR_DIR="${RUN_HOST}/supervisor"
DATA_HOST="${RUN_HOST}/data"
DATA_CONTAINER="${RUN_CONTAINER}/data"
MODEL_CONTAINER="${MODEL_CONTAINER:-${CONTAINER_PROJECT_ROOT}/exports/llin-qwen3.6-27b-grpo-step120-hf-20260813}"
APPROVED_SOURCE="${APPROVED_SOURCE:-${CONTAINER_PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_readiness_audit_20260822-04/private/mixed_approved_candidates.sensitive.parquet}"
AUDIT_SUMMARY="${AUDIT_SUMMARY:-${CONTAINER_PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_readiness_audit_20260822-04/safe_summary.json}"
VAL_FILE="${VAL_FILE:-${CONTAINER_PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset/boss_pi_val.parquet}"
CANONICAL_FILE="${DATA_CONTAINER}/train21.sensitive.parquet"
TRAIN_FILE="${DATA_CONTAINER}/train21x4.sensitive.parquet"
SAFE_SUMMARY="${DATA_CONTAINER}/train21x4.safe.json"
RAY_ADDRESS="192.168.202.5:36379"

mkdir -p "${SUPERVISOR_DIR}" "${DATA_HOST}"
exec 9>"${HOST_PROJECT_ROOT}/runs/.v15-mixed21-four-exposure.lock"
if ! flock -n 9; then
  printf 'another mixed21 four-exposure supervisor holds the lock\n' >&2
  exit 9
fi
if [[ -e "${RUN_HOST}/checkpoints" ]]; then
  printf 'refusing to reuse a run directory that already has checkpoints\n' >&2
  exit 10
fi
printf '%s\n' "$$" > "${SUPERVISOR_DIR}/supervisor.pid"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/started_at"

ray_started=false
cleanup() {
  local exit_status=$?
  if [[ "${ray_started}" == "true" ]]; then
    set +e
    docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
      "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    date --iso-8601=seconds > "${SUPERVISOR_DIR}/ray_cleanup_finished_at"
    set -e
  fi
  if (( exit_status != 0 )) && [[ ! -f "${SUPERVISOR_DIR}/exit_code" ]]; then
    printf '%s\n' "${exit_status}" > "${SUPERVISOR_DIR}/exit_code"
    printf 'failed\n' > "${SUPERVISOR_DIR}/state"
    date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
  fi
}
trap cleanup EXIT

npu_process_pattern='^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|[[:space:]]*[[:alnum:]_]'
assert_idle() {
  local local_npu remote_npu available_kb mlocked_kb
  local_npu="$(npu-smi info)"
  remote_npu="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" npu-smi info)"
  if [[ -z "${local_npu}" || -z "${remote_npu}" ]] \
      || grep -Eq "${npu_process_pattern}" <<< "${local_npu}" \
      || grep -Eq "${npu_process_pattern}" <<< "${remote_npu}"; then
    printf 'trainer or rollout NPU is not idle\n' >&2
    return 1
  fi
  available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  mlocked_kb="$(awk '/^Mlocked:/ {print $2}' /proc/meminfo)"
  if (( available_kb < 536870912 || mlocked_kb > 134217728 )); then
    printf 'trainer host memory gate failed: available=%s mlocked=%s\n' "${available_kb}" "${mlocked_kb}" >&2
    return 1
  fi
}

container_exec() {
  docker exec "${TRAINER_CONTAINER}" bash -lc "$1"
}

printf 'building_exact_approved_schedule\n' > "${SUPERVISOR_DIR}/state"
container_exec "python3 '${CONTAINER_PROJECT_ROOT}/scripts/prepare_v15_mixed21_training.py' build \
  --source '${APPROVED_SOURCE}' \
  --audit-summary '${AUDIT_SUMMARY}' \
  --canonical '${CANONICAL_FILE}' \
  --schedule '${TRAIN_FILE}' \
  --validation '${VAL_FILE}' \
  --safe-summary '${SAFE_SUMMARY}' \
  --seed 20260822" > "${SUPERVISOR_DIR}/exact_data_build.safe.json"
container_exec "python3 '${CONTAINER_PROJECT_ROOT}/scripts/prepare_v15_mixed21_training.py' validate \
  --source '${APPROVED_SOURCE}' \
  --audit-summary '${AUDIT_SUMMARY}' \
  --canonical '${CANONICAL_FILE}' \
  --schedule '${TRAIN_FILE}' \
  --validation '${VAL_FILE}' \
  --safe-summary '${SAFE_SUMMARY}'" > "${SUPERVISOR_DIR}/exact_data_validation.safe.json"

printf 'syncing_private_runtime_assets\n' > "${SUPERVISOR_DIR}/state"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "mkdir -p '${DATA_HOST}'"
scp -p "${DATA_HOST}/train21.sensitive.parquet" \
  "${DATA_HOST}/train21x4.sensitive.parquet" \
  "${DATA_HOST}/train21x4.safe.json" \
  "root@${ROLLOUT_HOST}:${DATA_HOST}/" > "${SUPERVISOR_DIR}/private_data_sync.log" 2>&1

printf 'validating_runtime_and_model\n' > "${SUPERVISOR_DIR}/state"
runtime_preflight="python3 '${CONTAINER_PROJECT_ROOT}/scripts/pi_runtime_preflight.py' \
  --dataset '${TRAIN_FILE}' --sandbox-root /pi_sandbox \
  --reward-path '${CONTAINER_PROJECT_ROOT}/llin_verl/pi_reward.py' \
  --reward-function compute_score_strict_correctness_v3"
container_exec "${runtime_preflight}" > "${SUPERVISOR_DIR}/trainer_runtime_preflight.safe.json"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${runtime_preflight}\"" \
  > "${SUPERVISOR_DIR}/rollout_runtime_preflight.safe.json"
model_files="test -s '${MODEL_CONTAINER}/config.json' && test -s '${MODEL_CONTAINER}/model.safetensors.index.json'"
container_exec "${model_files}"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${model_files}\""
local_model_hash="$(container_exec "sha256sum '${MODEL_CONTAINER}/config.json' '${MODEL_CONTAINER}/model.safetensors.index.json'")"
remote_model_hash="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"sha256sum '${MODEL_CONTAINER}/config.json' '${MODEL_CONTAINER}/model.safetensors.index.json'\"")"
if [[ "${local_model_hash}" != "${remote_model_hash}" ]]; then
  printf 'm05/m06 Step120 HF model manifests differ\n' >&2
  exit 11
fi
printf '%s\n' "${local_model_hash}" > "${SUPERVISOR_DIR}/model_manifest.sha256"

printf 'checking_resources\n' > "${SUPERVISOR_DIR}/state"
for attempt in 1 2 3; do
  assert_idle
  printf '%s\n' "${attempt}" > "${SUPERVISOR_DIR}/consecutive_idle_checks"
  if (( attempt < 3 )); then sleep 10; fi
done

printf 'starting_ray\n' > "${SUPERVISOR_DIR}/state"
set +e
container_exec 'ray stop --force' > "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" >> "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
set -e
container_exec "RAY_HEAD_PORT=36379 bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_qwen38_smoke_m05.sh'" \
  > "${SUPERVISOR_DIR}/ray_m05.log" 2>&1
ray_started=true
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"RAY_HEAD_ADDRESS='${RAY_ADDRESS}' bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_qwen38_smoke_m06.sh'\"" \
  > "${SUPERVISOR_DIR}/ray_m06.log" 2>&1
container_exec "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_qwen38_smoke_ray_cluster.py' --address '${RAY_ADDRESS}'" \
  > "${SUPERVISOR_DIR}/ray_cluster.safe.json"
container_exec "RAY_ADDRESS='${RAY_ADDRESS}' ROLLOUT_RANKS=16 python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_hccl_fanout.py'" \
  > "${SUPERVISOR_DIR}/hccl_fanout_1x16.safe.json" 2>&1

run_phase() {
  local phase="$1" source_checkpoint="$2" log_file="$3"
  container_exec "RAY_ADDRESS='${RAY_ADDRESS}' \
    RUN_NAME='${RUN_NAME}' OUTPUT_DIR='${RUN_CONTAINER}' DATA_DIR='${DATA_CONTAINER}' \
    MODEL_PATH='${MODEL_CONTAINER}' APPROVED_SOURCE='${APPROVED_SOURCE}' AUDIT_SUMMARY='${AUDIT_SUMMARY}' \
    VAL_FILE='${VAL_FILE}' PHASE='${phase}' SOURCE_CHECKPOINT='${source_checkpoint}' \
    bash '${CONTAINER_PROJECT_ROOT}/scripts/run_pi_v15_mixed21_4x_strict_kl.sh'" \
    > "${log_file}" 2>&1
}

printf 'canary_training\n' > "${SUPERVISOR_DIR}/state"
set +e
run_phase canary '' "${SUPERVISOR_DIR}/canary_driver.log"
canary_exit=$?
set -e
if (( canary_exit != 0 )); then
  printf '%s\n' "${canary_exit}" > "${SUPERVISOR_DIR}/exit_code"
  printf 'stopped_canary_training_failure\n' > "${SUPERVISOR_DIR}/state"
  date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
  exit "${canary_exit}"
fi

printf 'canary_gate\n' > "${SUPERVISOR_DIR}/state"
canary_steps=()
for step in 1 2 3 4 5; do canary_steps+=(--expected-step "${step}"); done
set +e
container_exec "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_v15_mixed21_canary.py' \
  --driver-log '${RUN_CONTAINER}/supervisor/canary_driver.log' \
  --baseline-step 0 --mode canary \
  ${canary_steps[*]} \
  --output '${RUN_CONTAINER}/supervisor/canary_gate.safe.json'" \
  > "${SUPERVISOR_DIR}/canary_gate_stdout.log" 2>&1
gate_exit=$?
set -e
if (( gate_exit != 0 )); then
  printf '%s\n' "${gate_exit}" > "${SUPERVISOR_DIR}/exit_code"
  printf 'stopped_canary_gate\n' > "${SUPERVISOR_DIR}/state"
  date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
  exit 85
fi

printf 'full_training\n' > "${SUPERVISOR_DIR}/state"
run_phase full "${RUN_CONTAINER}/checkpoints/global_step_5" "${SUPERVISOR_DIR}/full_driver.log"

printf 'final_gate\n' > "${SUPERVISOR_DIR}/state"
all_steps=()
for step in 1 2 3 4 5 42; do all_steps+=(--expected-step "${step}"); done
container_exec "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_v15_mixed21_canary.py' \
  --driver-log '${RUN_CONTAINER}/supervisor/canary_driver.log' \
  --driver-log '${RUN_CONTAINER}/supervisor/full_driver.log' \
  --baseline-step 0 --mode final \
  ${all_steps[*]} \
  --output '${RUN_CONTAINER}/supervisor/final_gate.safe.json'" \
  > "${SUPERVISOR_DIR}/final_gate_stdout.log" 2>&1

docker exec -i "${TRAINER_CONTAINER}" python3 - "${RUN_CONTAINER}" <<'PY' > "${SUPERVISOR_DIR}/final_checkpoint_gate.safe.json"
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "checkpoints"
observed = sorted(int(path.name.rsplit("_", 1)[1]) for path in root.glob("global_step_*") if path.is_dir())
required = {1, 2, 3, 4, 5, 42}
final_actor = root / "global_step_42" / "actor"
manifest = final_actor / "ckpt_contents.json"
valid = required.issubset(observed) and manifest.is_file()
print(json.dumps({
    "contract": "v15-mixed21-four-exposure-final-checkpoint-v1",
    "valid": valid,
    "required_checkpoint_steps": sorted(required),
    "observed_checkpoint_steps": observed,
    "final_actor_manifest_present": manifest.is_file(),
    "contains_sensitive_data": False,
}))
raise SystemExit(0 if valid else 1)
PY

printf '0\n' > "${SUPERVISOR_DIR}/exit_code"
printf 'complete\n' > "${SUPERVISOR_DIR}/state"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
