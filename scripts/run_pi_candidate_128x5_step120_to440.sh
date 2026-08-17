#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_ROOT}/runs/llin-grpo-candidate-pool-161-20260817-01/split-128-33-seed20260817}"
TRAIN_FILE="${TRAIN_FILE:-${SPLIT_DIR}/train128.sensitive.parquet}"
TEST_FILE="${TEST_FILE:-${SPLIT_DIR}/test33.sensitive.parquet}"
SAFE_SUMMARY="${SAFE_SUMMARY:-${SPLIT_DIR}/split.safe.json}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
RUN_NAME="${RUN_NAME:-llin-grpo-candidate128-5epoch-step120-to440-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

START_POLICY_STEP=120
TRAIN_TASKS=128
TEST_TASKS=33
TRAIN_EPOCHS=5
GROUPS_PER_STEP=2
RESPONSES_PER_GROUP=8
GROUPS_PER_EPOCH="${TRAIN_TASKS}"
STEPS_PER_EPOCH="$((GROUPS_PER_EPOCH / GROUPS_PER_STEP))"
NEW_TRAINING_STEPS="$((TRAIN_EPOCHS * STEPS_PER_EPOCH))"
FINAL_POLICY_STEP="$((START_POLICY_STEP + NEW_TRAINING_STEPS))"
SAVE_FREQ=40
AGENT_TIMEOUT_SECONDS=1800

if (( TRAIN_TASKS % GROUPS_PER_STEP != 0 )); then
  printf 'train task count must be divisible by groups per step\n' >&2
  exit 2
fi
if (( NEW_TRAINING_STEPS != 320 || FINAL_POLICY_STEP != 440 )); then
  printf 'unexpected five-epoch step contract: new=%s final=%s\n' \
    "${NEW_TRAINING_STEPS}" "${FINAL_POLICY_STEP}" >&2
  exit 2
fi

python3 "${PROJECT_ROOT}/scripts/split_grpo_candidate_pool.py" validate \
  --train "${TRAIN_FILE}" \
  --test "${TEST_FILE}" \
  --safe-summary "${SAFE_SUMMARY}" \
  --expected-rows 161 \
  --expected-train-rows "${TRAIN_TASKS}" \
  --sandbox-root /pi_sandbox

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/candidate_training_contract.txt" <<EOF
contract=llin-grpo-candidate128-five-epoch-step120-v1
source_checkpoint=${SOURCE_CHECKPOINT}
start_policy_step=${START_POLICY_STEP}
train_tasks=${TRAIN_TASKS}
test_tasks=${TEST_TASKS}
train_epochs=${TRAIN_EPOCHS}
groups_per_epoch=${GROUPS_PER_EPOCH}
steps_per_epoch=${STEPS_PER_EPOCH}
new_rollout_groups=$((TRAIN_TASKS * TRAIN_EPOCHS))
new_optimizer_steps=${NEW_TRAINING_STEPS}
final_policy_step=${FINAL_POLICY_STEP}
groups_per_step=${GROUPS_PER_STEP}
responses_per_group=${RESPONSES_PER_GROUP}
save_every_steps=${SAVE_FREQ}
kept_checkpoints=2
agent_timeout_seconds=${AGENT_TIMEOUT_SECONDS}
tool_timeout_seconds=${AGENT_TIMEOUT_SECONDS}
validation=final_only_test33
owner_authorized_training=true
promotion_allowed=false
checkpoint_payload=model,extra
optimizer_checkpoint_saved=false
EOF

DATA_DIR="${SPLIT_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${TEST_FILE}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
START_POLICY_STEP="${START_POLICY_STEP}" \
NEW_TRAINING_STEPS="${NEW_TRAINING_STEPS}" \
LOAD_OPTIMIZER_STATE=false \
SAVE_OPTIMIZER_STATE=false \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ="${FINAL_POLICY_STEP}" \
MAX_ACTOR_CKPT_TO_KEEP=2 \
LOG_VAL_GENERATIONS="${TEST_TASKS}" \
VALIDATION_LABEL=final_only_candidate_test33 \
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS}" \
DATA_SEED=20260817 \
DATA_PREFLIGHT_MODE=prevalidated \
WORKSPACE_TOOL_CONFIG_PATH="${PROJECT_ROOT}/configs/pi_workspace_tools_relaxed1800.yaml" \
bash "${PROJECT_ROOT}/scripts/run_pi_banded_2x8_resume.sh"
