#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full277_20260804/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/boss_pi_train.parquet}"
EVAL_FILE="${EVAL_FILE:-${DATA_DIR}/boss_pi_val.parquet}"
RUN_NAME="${RUN_NAME:-llin-pi-formal-frozen-baseline-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  printf 'Frozen baseline train file not found: %s\n' "${TRAIN_FILE}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'Frozen baseline evaluation file not found: %s\n' "${EVAL_FILE}" >&2
  exit 2
fi

# 25 tool-result turns require one additional assistant turn for the final
# answer.  This is intentionally 26/25, not the historical 25/24 boundary.
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=-1 \
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-26}" \
MAX_USER_TURNS="${MAX_USER_TURNS:-25}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${EVAL_FILE}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.actor.megatron.forward_only=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.grad_offload=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.rollout_data_dir=null \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  "$@"
