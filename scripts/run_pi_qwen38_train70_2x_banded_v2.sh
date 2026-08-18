#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
POOL_DIR="${POOL_DIR:-${PROJECT_ROOT}/runs/llin-qwen38-grpo-train70-2x-20260818-01/data}"
CANONICAL_FILE="${CANONICAL_FILE:-${POOL_DIR}/train70.sensitive.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${POOL_DIR}/train70x2.sensitive.parquet}"
SAFE_SUMMARY="${SAFE_SUMMARY:-${POOL_DIR}/train70x2.safe.json}"
RUN_NAME="${RUN_NAME:-llin-qwen38-grpo-train70-2x-banded-v2-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

TRAIN_TASKS=70
EXPOSURES_PER_TASK=2
GROUPS_PER_STEP=2
RESPONSES_PER_GROUP=8
TOTAL_ROLLOUT_GROUPS="$((TRAIN_TASKS * EXPOSURES_PER_TASK))"
TOTAL_TRAINING_STEPS="$((TOTAL_ROLLOUT_GROUPS / GROUPS_PER_STEP))"
MAX_PROMPT_TOKENS=4096
MAX_RESPONSE_TOKENS=49152
MAX_CONTEXT_TOKENS=53248
AGENT_TIMEOUT_SECONDS=1800
PREWARM_GROUPS=4
MAX_QUEUE_GROUPS=6
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * RESPONSES_PER_GROUP * MAX_CONTEXT_TOKENS))"
EMBEDDING_WEIGHT_MIB=2425
MIN_WEIGHT_BUCKET_MB=2560
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB:-2560}"

if (( TOTAL_ROLLOUT_GROUPS != 140 || TOTAL_TRAINING_STEPS != 70 )); then
  printf 'unexpected Qwen3.8 train70 shape\n' >&2
  exit 2
fi
if (( WEIGHT_BUCKET_MB < MIN_WEIGHT_BUCKET_MB )); then
  printf 'weight sync bucket %s MiB cannot safely hold the %s MiB Qwen3.8 embedding tensor; require at least %s MiB\n' \
    "${WEIGHT_BUCKET_MB}" "${EMBEDDING_WEIGHT_MIB}" "${MIN_WEIGHT_BUCKET_MB}" >&2
  exit 2
fi
python3 "${PROJECT_ROOT}/scripts/assemble_qwen38_train70.py" validate \
  --canonical "${CANONICAL_FILE}" \
  --schedule "${TRAIN_FILE}" \
  --safe-summary "${SAFE_SUMMARY}"

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/training_contract.txt" <<EOF
contract=llin-qwen38-grpo-train70-two-exposure-formal-v1
model=qwen38-27b-native-hf
initialization=qwen38_hf_base
qwen36_checkpoint_reused=false
reward=banded-v2-strict-table-v1
owner_authorized_all_70=true
strict_baseline_variance_tasks=20
train_tasks=${TRAIN_TASKS}
exposures_per_task=${EXPOSURES_PER_TASK}
rollout_groups=${TOTAL_ROLLOUT_GROUPS}
responses_per_group=${RESPONSES_PER_GROUP}
groups_per_optimizer_step=${GROUPS_PER_STEP}
optimizer_steps=${TOTAL_TRAINING_STEPS}
learning_rate=1e-7
reasoning_effort=medium
max_prompt_tokens=${MAX_PROMPT_TOKENS}
max_response_tokens=${MAX_RESPONSE_TOKENS}
max_context_tokens=${MAX_CONTEXT_TOKENS}
trajectory_timeout_seconds=${AGENT_TIMEOUT_SECONDS}
timeout_scope=entire_pi_agent_run_including_vllm_admission_generation_and_tools
temperature=1.0
top_p=0.95
top_k=20
trainer_npus=16
trainer_topology=tp4_pp2_cp2
rollout_npus=16
rollout_topology=tp4_dp4
rollout_max_num_seqs_per_replica=16
embedding_weight_mib=${EMBEDDING_WEIGHT_MIB}
weight_sync_bucket_mib=${WEIGHT_BUCKET_MB}
optimizer_offload=device_side
checkpoint_frequency=${TOTAL_TRAINING_STEPS}
kept_checkpoints=1
checkpoint_payload=model,extra
optimizer_checkpoint_saved=false
validation=disabled_no_heldout_rows
promotion_allowed=false
EOF

PI_REWARD_MODE=banded_v2 \
MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
SAVE_FREQ="${TOTAL_TRAINING_STEPS}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES="${RESPONSES_PER_GROUP}" \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD=2.0 \
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
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}" \
OPTIMIZER_CPU_OFFLOAD=false \
ENGINE_OPTIMIZER_OFFLOAD=false \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.shuffle=false \
  data.seed=20260818 \
  +data.apply_chat_template_kwargs.reasoning_effort=medium \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.actor.optim.lr=1e-7 \
  reward.custom_reward_function.name=compute_score_banded_v2 \
  trainer.project_name=llin-qwen38-verl-grpo \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  async_training.use_trainer_do_validate=false \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'
