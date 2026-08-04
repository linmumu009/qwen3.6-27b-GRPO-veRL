#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full277_20260804/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/boss_pi_train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/boss_pi_val.parquet}"
RUN_NAME="${RUN_NAME:-llin-pi-formal-grpo-4of4-50step-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
GROUPS_PER_STEP="${GROUPS_PER_STEP:-4}"
EVAL_FREQ="${EVAL_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
PREWARM_GROUPS="${PREWARM_GROUPS:-8}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-1.0}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-49152}"
MAX_QUEUE_GROUPS="${MAX_QUEUE_GROUPS:-8}"
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS:-$((MAX_QUEUE_GROUPS * 4 * MAX_CONTEXT_TOKENS))}"

# A formal run may only consume the source-joined, reviewed boss contract.
# The old formal_pi_v2_* dataset is intentionally rejected here.
python3 "${PROJECT_ROOT}/scripts/check_boss_alignment_contract.py" \
  --data-dir "${DATA_DIR}"

for path in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Formal PI data file not found: %s\n' "${path}" >&2
    exit 2
  fi
done
if (( TOTAL_TRAINING_STEPS <= 0 || GROUPS_PER_STEP != 4 )); then
  printf 'Formal run requires positive steps and exactly 4 groups per update\n' >&2
  exit 2
fi
if (( EVAL_FREQ <= 0 || SAVE_FREQ <= 0 )); then
  printf 'Evaluation and checkpoint frequencies must be positive\n' >&2
  exit 2
fi
if (( PREWARM_GROUPS > MAX_QUEUE_GROUPS )); then
  printf 'Prewarm groups (%s) exceed queue group bound (%s)\n' \
    "${PREWARM_GROUPS}" "${MAX_QUEUE_GROUPS}" >&2
  exit 2
fi

python3 "${PROJECT_ROOT}/scripts/check_formal_data_on_ray.py" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --ray-address "${RAY_ADDRESS:-192.168.202.5:26379}"

# Exact 4->4 sampling: no fastest-K surplus candidates are created. The test
# split remains sealed; all reviewed train tasks and the source-isolated
# validation split enter this run. Validation is greedy n=1 every EVAL_FREQ.
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="$((TOTAL_TRAINING_STEPS * GROUPS_PER_STEP))" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
SAVE_FREQ="${SAVE_FREQ}" \
FASTEST_K=4 \
OVERSAMPLE_CANDIDATES=4 \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD="${STALENESS_THRESHOLD}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=45056 \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.val_before_train=False \
  trainer.test_freq="${EVAL_FREQ}" \
  trainer.log_val_generations=20 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep=1 \
  async_training.use_trainer_do_validate=False \
  "$@"
