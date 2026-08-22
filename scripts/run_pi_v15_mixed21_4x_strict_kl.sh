#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/exports/llin-qwen3.6-27b-grpo-step120-hf-20260813}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train21x4.sensitive.parquet}"
CANONICAL_FILE="${CANONICAL_FILE:-${DATA_DIR}/train21.sensitive.parquet}"
SAFE_SUMMARY="${SAFE_SUMMARY:-${DATA_DIR}/train21x4.safe.json}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset/boss_pi_val.parquet}"
PHASE="${PHASE:-canary}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"

TRAIN_TASKS=21
EXPOSURES_PER_TASK=4
RESPONSES_PER_GROUP=8
GROUPS_PER_STEP=2
TOTAL_ROLLOUT_GROUPS="$((TRAIN_TASKS * EXPOSURES_PER_TASK))"
TOTAL_TRAINING_STEPS="$((TOTAL_ROLLOUT_GROUPS / GROUPS_PER_STEP))"
CANARY_STEPS=5
CANARY_GROUPS="$((CANARY_STEPS * GROUPS_PER_STEP))"
MAX_PROMPT_TOKENS=4096
MAX_RESPONSE_TOKENS=90112
MAX_CONTEXT_TOKENS=94208
AGENT_TIMEOUT_SECONDS=1800
MAX_QUEUE_GROUPS=2
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * RESPONSES_PER_GROUP * MAX_CONTEXT_TOKENS))"
LEARNING_RATE="${LEARNING_RATE:-5e-8}"
KL_COEF="${KL_COEF:-0.001}"

case "${PHASE}" in
  canary)
    TARGET_TRAINING_STEPS="${CANARY_STEPS}"
    TARGET_ROLLOUT_GROUPS="${CANARY_GROUPS}"
    SAVE_FREQ=1
    TEST_FREQ=1
    VAL_BEFORE_TRAIN=true
    RESUME_MODE=disable
    CHECKPOINT_LOAD_CONTENTS='[model,extra]'
    ;;
  full)
    if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
      printf 'full phase requires the verified canary checkpoint, got: %s\n' "${SOURCE_CHECKPOINT}" >&2
      exit 2
    fi
    TARGET_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}"
    TARGET_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}"
    SAVE_FREQ="${TOTAL_TRAINING_STEPS}"
    TEST_FREQ="${TOTAL_TRAINING_STEPS}"
    VAL_BEFORE_TRAIN=false
    RESUME_MODE=resume_path
    CHECKPOINT_LOAD_CONTENTS='[model,optimizer,extra]'
    ;;
  *)
    printf 'PHASE must be canary or full\n' >&2
    exit 2
    ;;
esac

if (( TOTAL_ROLLOUT_GROUPS != 84 || TOTAL_TRAINING_STEPS != 42 )); then
  printf 'unexpected mixed21 four-exposure shape\n' >&2
  exit 2
fi
python3 "${PROJECT_ROOT}/scripts/prepare_v15_mixed21_training.py" validate \
  --source "${APPROVED_SOURCE:?APPROVED_SOURCE is required}" \
  --audit-summary "${AUDIT_SUMMARY:?AUDIT_SUMMARY is required}" \
  --canonical "${CANONICAL_FILE}" \
  --schedule "${TRAIN_FILE}" \
  --validation "${VAL_FILE}" \
  --safe-summary "${SAFE_SUMMARY}"
python3 "${PROJECT_ROOT}/scripts/patch_verl_grpo_strict_variance_gate.py" \
  --trainer /verl/verl/experimental/separation/ray_trainer.py

mkdir -p "${OUTPUT_DIR}/supervisor"
cat > "${OUTPUT_DIR}/supervisor/${PHASE}_training_contract.txt" <<EOF
contract=v15-mixed21-four-exposure-strict-gated-v1
phase=${PHASE}
model_start=qwen3.6_step120_hf_export
model_path=${MODEL_PATH}
reference_model=frozen_same_step120_hf_export
reference_kl=fixed_in_reward
kl_coef=${KL_COEF}
reward_contract=strict-correctness-gated-v3
scalar_reward=strict_acc_binary
process_reward_applied=0
uniform_correctness_group_advantage=zero
uniform_only_optimizer_batch=skipped
train_tasks=${TRAIN_TASKS}
exposures_per_task=${EXPOSURES_PER_TASK}
total_rollout_groups=${TOTAL_ROLLOUT_GROUPS}
total_online_trajectories=$((TOTAL_ROLLOUT_GROUPS * RESPONSES_PER_GROUP))
total_optimizer_steps=${TOTAL_TRAINING_STEPS}
target_rollout_groups=${TARGET_ROLLOUT_GROUPS}
target_optimizer_steps=${TARGET_TRAINING_STEPS}
canary_steps=${CANARY_STEPS}
responses_per_group=${RESPONSES_PER_GROUP}
groups_per_step=${GROUPS_PER_STEP}
staleness_threshold=0
trainer_npus=16
trainer_topology=tp4_pp2_cp2
rollout_npus=16
rollout_topology=tp4_dp4
learning_rate=${LEARNING_RATE}
max_context_tokens=${MAX_CONTEXT_TOKENS}
trajectory_timeout_seconds=${AGENT_TIMEOUT_SECONDS}
validation_file_is_disjoint=true
promotion_allowed=false
EOF

resume_args=(trainer.resume_mode="${RESUME_MODE}")
if [[ "${PHASE}" == "full" ]]; then
  resume_args+=(trainer.resume_from_path="${SOURCE_CHECKPOINT}")
  resume_args+=(trainer.del_local_ckpt_after_load=False)
fi

PI_REWARD_MODE=banded_v2 \
MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${TARGET_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TARGET_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
SAVE_FREQ="${SAVE_FREQ}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES="${RESPONSES_PER_GROUP}" \
PREWARM_GROUPS=2 \
STALENESS_THRESHOLD=0 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS}" \
MAX_PARALLEL_TOOL_CALLS=4 \
MAX_TOOL_RESPONSE_CHARS=32768 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=16 \
AGENT_WORKERS=12 \
CONCURRENT_SAMPLES_PER_REPLICA=6 \
WEIGHT_BUCKET_MB=2560 \
OPTIMIZER_CPU_OFFLOAD=false \
ENGINE_OPTIMIZER_OFFLOAD=false \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.shuffle=false \
  data.seed=20260822 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools_relaxed1800.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  reward.custom_reward_function.name=compute_score_strict_correctness_v3 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.project_name=llin-qwen36-verl-grpo \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.log_val_generations=20 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep=6 \
  async_training.use_trainer_do_validate=false \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  "actor_rollout_ref.actor.checkpoint.load_contents=${CHECKPOINT_LOAD_CONTENTS}" \
  "${resume_args[@]}"
