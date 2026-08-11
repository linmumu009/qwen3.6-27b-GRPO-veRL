#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/boss_pi_train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/boss_pi_val.parquet}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
START_POLICY_STEP="${START_POLICY_STEP:?START_POLICY_STEP is required}"
NEW_TRAINING_STEPS="${NEW_TRAINING_STEPS:?NEW_TRAINING_STEPS is required}"
FINAL_POLICY_STEP="$((START_POLICY_STEP + NEW_TRAINING_STEPS))"
RUN_NAME="${RUN_NAME:-llin-banded-v1-2x8-step${START_POLICY_STEP}-to-${FINAL_POLICY_STEP}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

GROUPS_PER_STEP=2
RESPONSES_PER_GROUP=8
PREWARM_GROUPS=4
MAX_QUEUE_GROUPS=4
MAX_CONTEXT_TOKENS=49152
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * RESPONSES_PER_GROUP * MAX_CONTEXT_TOKENS))"
TOTAL_ROLLOUT_GROUPS="$((NEW_TRAINING_STEPS * GROUPS_PER_STEP + PREWARM_GROUPS))"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
LOAD_OPTIMIZER_STATE="${LOAD_OPTIMIZER_STATE:-true}"

case "${LOAD_OPTIMIZER_STATE}" in
  true)
    CHECKPOINT_LOAD_CONTENTS="[model,optimizer,extra]"
    OPTIMIZER_STATE="resume_from_source"
    ;;
  false)
    CHECKPOINT_LOAD_CONTENTS="[model,extra]"
    OPTIMIZER_STATE="reset_hybrid_cpu_offload_resume_workaround"
    ;;
  *)
    printf 'LOAD_OPTIMIZER_STATE must be true or false, got: %s\n' \
      "${LOAD_OPTIMIZER_STATE}" >&2
    exit 2
    ;;
esac

if (( START_POLICY_STEP < 0 || NEW_TRAINING_STEPS <= 0 )); then
  printf 'Invalid resume interval: start=%s new_steps=%s\n' \
    "${START_POLICY_STEP}" "${NEW_TRAINING_STEPS}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
  printf 'Source checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
python3 "${PROJECT_ROOT}/scripts/check_boss_alignment_contract.py" --data-dir "${DATA_DIR}"
python3 "${PROJECT_ROOT}/scripts/check_formal_data_on_ray.py" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --ray-address "${RAY_ADDRESS:-192.168.202.5:26379}"

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
reward=banded_v1
source_checkpoint=${SOURCE_CHECKPOINT}
start_policy_step=${START_POLICY_STEP}
new_training_steps=${NEW_TRAINING_STEPS}
final_policy_step=${FINAL_POLICY_STEP}
groups_per_update=${GROUPS_PER_STEP}
responses_per_group=${RESPONSES_PER_GROUP}
trajectories_per_update=16
all_responses_used=true
context_tokens=${MAX_CONTEXT_TOKENS}
validation=final_only_boss_val20
checkpoint=final_only_model_optimizer_extra
checkpoint_load_contents=${CHECKPOINT_LOAD_CONTENTS}
optimizer_state=${OPTIMIZER_STATE}
EOF

PI_REWARD_MODE=banded_v1 \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS="${FINAL_POLICY_STEP}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
SAVE_FREQ="${FINAL_POLICY_STEP}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES="${RESPONSES_PER_GROUP}" \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD=2.0 \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=45056 \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=24 \
AGENT_WORKERS=12 \
CONCURRENT_SAMPLES_PER_REPLICA=6 \
WEIGHT_BUCKET_MB=2560 \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  reward.custom_reward_function.name=compute_score_banded_v1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.val_before_train=False \
  trainer.test_freq="${FINAL_POLICY_STEP}" \
  trainer.log_val_generations=20 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.save_freq="${FINAL_POLICY_STEP}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
  trainer.del_local_ckpt_after_load=False \
  async_training.use_trainer_do_validate=False \
  "actor_rollout_ref.actor.checkpoint.load_contents=${CHECKPOINT_LOAD_CONTENTS}" \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]'
