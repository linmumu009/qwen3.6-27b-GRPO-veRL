#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_train236_20260811}"
EVAL_FILE="${EVAL_FILE:-${DATA_DIR}/repair_sft_replay.parquet}"
RUN_NAME="${RUN_NAME:-llin-native-repair-replay-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-26}"
MAX_USER_TURNS="${MAX_USER_TURNS:-25}"
EXPECTED_EVAL_ROWS="${EXPECTED_EVAL_ROWS:-16}"

if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'native attribution replay parquet missing: %s\n' "${EVAL_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=native_vs_step120_reward_behavior_attribution
model_source=original_hf_base_weights
evaluation_rows=${EXPECTED_EVAL_ROWS}
evaluation_split=train236_same_task_not_heldout
sampling=greedy_n1
system_tools=boss_exact
context_tokens=49152
max_tool_result_turns=${MAX_USER_TURNS}
max_assistant_turns=${MAX_ASSISTANT_TURNS}
forward_only=true
optimizer_initialized=false
checkpoint_saved=false
comparison_checkpoint=step120
promotion_allowed=false
EOF

unset LLIN_VAL_ONLY_FORCE_DIST_SYNC
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_FILE="${EVAL_FILE}" \
EVAL_FILE="${EVAL_FILE}" \
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS}" \
MAX_USER_TURNS="${MAX_USER_TURNS}" \
EXPECTED_EVAL_ROWS="${EXPECTED_EVAL_ROWS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=12 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path=null \
  'actor_rollout_ref.actor.checkpoint.load_contents=[]' \
  trainer.resume_mode=disable \
  trainer.val_only=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation"
