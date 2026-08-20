#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-qwen38-smoke-m05-20260817}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-qwen38-smoke-m06-20260817}"
RUN_NAME="${RUN_NAME:-llin-qwen38-grpo-train70-2x-banded-v2-20260818-01}"
POOL_DIR="${POOL_DIR:-${CONTAINER_PROJECT_ROOT}/runs/llin-qwen38-grpo-train70-2x-20260818-01/data}"
CANONICAL_FILE="${CANONICAL_FILE:-${POOL_DIR}/train70.sensitive.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${POOL_DIR}/train70x2.sensitive.parquet}"
SEALED_FILE="${SEALED_FILE:-}"
SAFE_SUMMARY="${SAFE_SUMMARY:-${POOL_DIR}/train70x2.safe.json}"
ASSEMBLER_SCRIPT="${ASSEMBLER_SCRIPT:-assemble_qwen38_train70.py}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-run_pi_qwen38_train70_2x_banded_v2.sh}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
MODEL_EXPORT_POLICY_STEP="${MODEL_EXPORT_POLICY_STEP:-}"
EXPECTED_CHECKPOINT_STEP="${EXPECTED_CHECKPOINT_STEP:-70}"
SUPERVISOR_DIR="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}-supervisor"
RAY_ADDRESS="192.168.202.5:36379"

mkdir -p "${SUPERVISOR_DIR}"
exec 9>"${HOST_PROJECT_ROOT}/runs/.qwen38-train70.lock"
if ! flock -n 9; then
  printf 'another Qwen3.8 train70 supervisor holds the lock\n' >&2
  exit 9
fi
printf '%s\n' "$$" > "${SUPERVISOR_DIR}/supervisor.pid"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/started_at"

ray_started=false
cleanup() {
  if [[ "${ray_started}" == "true" ]]; then
    set +e
    docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
      "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" >> "${SUPERVISOR_DIR}/ray_cleanup.log" 2>&1
    date --iso-8601=seconds > "${SUPERVISOR_DIR}/ray_cleanup_finished_at"
    set -e
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

printf 'validating_assets\n' > "${SUPERVISOR_DIR}/state"
validate="python3 '${CONTAINER_PROJECT_ROOT}/scripts/${ASSEMBLER_SCRIPT}' validate \
  --canonical '${CANONICAL_FILE}' \
  --schedule '${TRAIN_FILE}' \
  --safe-summary '${SAFE_SUMMARY}'"
if [[ -n "${SEALED_FILE}" ]]; then
  validate+=" --sealed '${SEALED_FILE}'"
fi
docker exec "${TRAINER_CONTAINER}" bash -lc "${validate}" > "${SUPERVISOR_DIR}/trainer_data_validation.json"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${validate}\"" > "${SUPERVISOR_DIR}/rollout_data_validation.json"
runtime_preflight="python3 '${CONTAINER_PROJECT_ROOT}/scripts/pi_runtime_preflight.py' \
  --dataset '${TRAIN_FILE}' --sandbox-root /pi_sandbox \
  --reward-path '${CONTAINER_PROJECT_ROOT}/llin_verl/pi_reward.py' \
  --reward-function compute_score_banded_v2"
docker exec "${TRAINER_CONTAINER}" bash -lc "${runtime_preflight}" \
  > "${SUPERVISOR_DIR}/trainer_runtime_preflight.json"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${runtime_preflight}\"" \
  > "${SUPERVISOR_DIR}/rollout_runtime_preflight.json"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_qwen38_model_compat.py' --reference-model /models/Qwen3.6-27B --candidate-model '${MODEL_PATH}' --output '${POOL_DIR}/trainer_model_compat.safe.json'"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_qwen38_model_compat.py' --reference-model /models/Qwen3.6-27B --candidate-model '${MODEL_PATH}' --output '${POOL_DIR}/rollout_model_compat.safe.json'\""
if [[ -n "${MODEL_EXPORT_POLICY_STEP}" ]]; then
  export_gate="python3 -c 'import json,pathlib; p=json.loads((pathlib.Path(\"${MODEL_PATH}\")/\"llin_export_manifest.json\").read_text()); assert (p.get(\"verification\") or {}).get(\"valid\") is True; assert \"global_step_${MODEL_EXPORT_POLICY_STEP}\" in str(p.get(\"actor_checkpoint\") or \"\")'"
  docker exec "${TRAINER_CONTAINER}" bash -lc "${export_gate}"
  ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
    "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"${export_gate}\""
fi

printf 'checking_resources\n' > "${SUPERVISOR_DIR}/state"
for attempt in 1 2 3; do
  assert_idle
  printf '%s\n' "${attempt}" > "${SUPERVISOR_DIR}/consecutive_idle_checks"
  if (( attempt < 3 )); then sleep 10; fi
done

printf 'starting_ray\n' > "${SUPERVISOR_DIR}/state"
set +e
docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' > "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc 'ray stop --force'" >> "${SUPERVISOR_DIR}/stale_ray_cleanup.log" 2>&1
set -e
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RAY_HEAD_PORT=36379 bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_qwen38_smoke_m05.sh'" \
  > "${SUPERVISOR_DIR}/ray_m05.log" 2>&1
ray_started=true
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"RAY_HEAD_ADDRESS='${RAY_ADDRESS}' bash '${CONTAINER_PROJECT_ROOT}/scripts/start_ray_qwen38_smoke_m06.sh'\"" \
  > "${SUPERVISOR_DIR}/ray_m06.log" 2>&1
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_qwen38_smoke_ray_cluster.py' --address '${RAY_ADDRESS}'" \
  > "${SUPERVISOR_DIR}/ray_cluster.safe.json"

printf 'checking_hccl_fanout\n' > "${SUPERVISOR_DIR}/state"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RAY_ADDRESS='${RAY_ADDRESS}' ROLLOUT_RANKS=16 python3 '${CONTAINER_PROJECT_ROOT}/scripts/check_hccl_fanout.py'" \
  > "${SUPERVISOR_DIR}/hccl_fanout_1x16.json" 2>&1

printf 'training\n' > "${SUPERVISOR_DIR}/state"
set +e
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "RAY_ADDRESS='${RAY_ADDRESS}' RUN_NAME='${RUN_NAME}' MODEL_PATH='${MODEL_PATH}' POOL_DIR='${POOL_DIR}' CANONICAL_FILE='${CANONICAL_FILE}' TRAIN_FILE='${TRAIN_FILE}' SEALED_FILE='${SEALED_FILE}' SAFE_SUMMARY='${SAFE_SUMMARY}' bash '${CONTAINER_PROJECT_ROOT}/scripts/${TRAINING_SCRIPT}'" \
  > "${SUPERVISOR_DIR}/training_launcher.log" 2>&1
training_exit=$?
set -e
if [[ "${training_exit}" == "0" ]]; then
  printf 'verifying_final_checkpoint\n' > "${SUPERVISOR_DIR}/state"
  set +e
  docker exec -i "${TRAINER_CONTAINER}" python3 - "${CONTAINER_PROJECT_ROOT}/runs/${RUN_NAME}" "${EXPECTED_CHECKPOINT_STEP}" <<'PY' \
    > "${SUPERVISOR_DIR}/final_checkpoint_gate.json" 2>&1
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
expected_step = int(sys.argv[2])
root = run / "checkpoints"
steps = sorted(path for path in root.glob("global_step_*") if path.is_dir())
expected = root / f"global_step_{expected_step}"
actor = expected / "actor"
manifest_path = actor / "ckpt_contents.json"
model_format = None
model_shards = 0
metadata_exists = False
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_entry = ((manifest.get("contents") or {}).get("model") or {})
    model_format = model_entry.get("format")
    relative = Path(str(model_entry.get("path") or ""))
    if relative.parts and not relative.is_absolute() and ".." not in relative.parts:
        model_dir = actor / relative
        model_shards = len(list(model_dir.glob("*.distcp")))
        metadata_exists = (model_dir / "metadata.json").is_file() or (model_dir / ".metadata").is_file()
valid = (
    steps == [expected]
    and model_format == "megatron_dist_checkpoint"
    and metadata_exists
    and model_shards > 0
)
print(json.dumps({
    "contract": "llin-qwen38-final-checkpoint-completion-gate-v1",
    "valid": valid,
    "observed_checkpoint_steps": [path.name for path in steps],
    "expected_step": expected_step,
    "model_format": model_format,
    "model_shards": model_shards,
    "metadata_exists": metadata_exists,
    "contains_server_paths": False,
}))
raise SystemExit(0 if valid else 1)
PY
  checkpoint_gate_exit=$?
  set -e
  if [[ "${checkpoint_gate_exit}" != "0" ]]; then
    training_exit=86
  fi
fi
printf '%s\n' "${training_exit}" > "${SUPERVISOR_DIR}/exit_code"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
if [[ "${training_exit}" == "0" ]]; then
  printf 'complete\n' > "${SUPERVISOR_DIR}/state"
else
  printf 'failed\n' > "${SUPERVISOR_DIR}/state"
fi
exit "${training_exit}"
