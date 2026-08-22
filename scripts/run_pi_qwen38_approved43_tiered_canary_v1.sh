#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_readiness_audit_20260822-05}"
APPROVED43="${APPROVED43:-${PACKAGE_ROOT}/private/grpo_approved43.sensitive.parquet}"
APPROVED43_MANIFEST="${APPROVED43_MANIFEST:-${PACKAGE_ROOT}/private/grpo_approved43_manifest.sensitive.jsonl}"
TASKS_FILE="${TASKS_FILE:?TASKS_FILE must be the frozen private 100-task JSONL used only for approved manifest indices}"
RAW100_FILE="${RAW100_FILE:-${PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/data/rollout_100.sensitive.parquet}"
RUN_NAME="${RUN_NAME:-llin-qwen38-approved43-tiered-v1-canary5-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${OUTPUT_DIR}/private/canary20.sensitive.parquet}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-${OUTPUT_DIR}/canary20.safe.json}"
SEALED_FILE="${SEALED_FILE:-${OUTPUT_DIR}/private/sealed8.sensitive.parquet}"
SEALED_SUMMARY="${SEALED_SUMMARY:-${OUTPUT_DIR}/sealed8.safe.json}"

# Data preparation and reward imports run before the base veRL launcher exports
# its Python path. Bind the isolated runtime explicitly so the canary never
# falls back to an unrelated site-package or the host checkout.
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"

EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f"
EXPECTED_APPROVED43_SHA256="d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
EXPECTED_MANIFEST_SHA256="1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"
TARGET_ACTUAL_OPTIMIZER_STEPS=5
MAX_NOMINAL_GROUPS=20
GROUPS_PER_STEP=2
RESPONSES_PER_GROUP=8
MAX_NOMINAL_BATCHES=10

[[ "$(sha256sum "${APPROVED43}" | awk '{print $1}')" == "${EXPECTED_APPROVED43_SHA256}" ]] || {
  printf 'approved43 parquet hash mismatch\n' >&2; exit 2;
}
[[ "$(sha256sum "${APPROVED43_MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] || {
  printf 'approved43 manifest hash mismatch\n' >&2; exit 2;
}
[[ "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA256}" ]] || {
  printf 'Qwen3.8 config hash mismatch\n' >&2; exit 2;
}
mapfile -t model_shards < <(printf '%s\n' "${MODEL_PATH}"/model-*-of-00018.safetensors | LC_ALL=C sort)
[[ "${#model_shards[@]}" == 18 ]] || { printf 'Qwen3.8 must have exactly 18 model shards\n' >&2; exit 2; }
observed_model_compound_sha256="$(LC_ALL=C sha256sum "${model_shards[@]}" | sha256sum | awk '{print $1}')"
[[ "${observed_model_compound_sha256}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]] || {
  printf 'Qwen3.8 18-shard compound hash mismatch\n' >&2; exit 2;
}

mkdir -p "${OUTPUT_DIR}/private" "${OUTPUT_DIR}/private_recovery" "${OUTPUT_DIR}/audit"
chmod 700 "${OUTPUT_DIR}/private" "${OUTPUT_DIR}/private_recovery" "${OUTPUT_DIR}/audit"
python3 "${PROJECT_ROOT}/scripts/prepare_qwen38_tiered_canary_data.py" \
  --approved43 "${APPROVED43}" \
  --manifest "${APPROVED43_MANIFEST}" \
  --tasks "${TASKS_FILE}" \
  --output "${TRAIN_FILE}" \
  --safe-summary "${TRAIN_SUMMARY}"
python3 "${PROJECT_ROOT}/scripts/prepare_qwen38_tiered_canary_sealed8.py" \
  --approved43 "${APPROVED43}" \
  --raw100 "${RAW100_FILE}" \
  --output "${SEALED_FILE}" \
  --safe-summary "${SEALED_SUMMARY}"

python3 - "${APPROVED43}" "${TRAIN_FILE}" "${SEALED_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq

approved, train, sealed = [pq.read_table(path).to_pylist() for path in sys.argv[1:]]
identity = lambda row: str((row.get("extra_info") or {}).get("instruction_sha256") or "")
answer_type = lambda row: str(((row.get("reward_model") or {}).get("ground_truth") or {}).get("answer_type") or "numeric").casefold()
approved_ids = {identity(row) for row in approved}
train_ids = [identity(row) for row in train]
sealed_ids = [identity(row) for row in sealed]
if len(approved) != 43 or len(approved_ids) != 43:
    raise SystemExit("approved43 membership shape mismatch")
if len(train) != 20 or len(set(train_ids)) != 20 or not set(train_ids) <= approved_ids:
    raise SystemExit("canary schedule is not 20 unique approved43 groups")
if len(sealed) != 8 or len(set(sealed_ids)) != 8 or approved_ids & set(sealed_ids):
    raise SystemExit("sealed evaluation must be eight unique non-approved tasks")
counts = {kind: sum(answer_type(row) == kind for row in sealed) for kind in ("numeric", "table")}
if counts != {"numeric": 4, "table": 4}:
    raise SystemExit(f"sealed answer-type shape mismatch: {counts}")
if any((row.get("extra_info") or {}).get("training_allowed") is not False for row in sealed):
    raise SystemExit("sealed task is not explicitly training-disabled")
PY

python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" \
  --rollouter "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_rollouter.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --tool-agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/tool_agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_hard_gate_resampling.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_grpo_strict_variance_gate.py" \
  --trainer "${VERL_ROOT}/verl/experimental/separation/ray_trainer.py" \
  --fully-async-trainer "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_trainer.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_canary_rollout_audit.py" \
  --trainer "${VERL_ROOT}/verl/experimental/separation/ray_trainer.py"

cat > "${OUTPUT_DIR}/audit/canary_contract.safe.txt" <<EOF
contract=qwen38-approved43-tiered-query-cost-actual-update-canary-v1
actor_initialization=${MODEL_PATH}
reference_initialization=${MODEL_PATH}
qwen36_or_historical_checkpoint_reused=false
model_config_sha256=${EXPECTED_CONFIG_SHA256}
model_18shard_compound_sha256_method=LC_ALL=C_sha256sum_absolute_sorted_glob_lines_then_sha256
model_18shard_compound_sha256=${EXPECTED_MODEL_COMPOUND_SHA256}
approved43_parquet_sha256=${EXPECTED_APPROVED43_SHA256}
approved43_manifest_sha256=${EXPECTED_MANIFEST_SHA256}
reward=compute_score_tiered_query_cost_v1
reward_scope=one_scalar_after_complete_multiturn_trajectory
algorithm_use_kl_in_reward=false
actor_use_kl_loss=true
actor_kl_loss_coef=0.001
actor_kl_loss_type=low_var_kl
hard_staleness=exact_actual_policy_version
target_actual_optimizer_steps=${TARGET_ACTUAL_OPTIMIZER_STEPS}
maximum_nominal_groups=${MAX_NOMINAL_GROUPS}
groups_per_nominal_batch=${GROUPS_PER_STEP}
responses_per_group=${RESPONSES_PER_GROUP}
unknown_behavior=resample_then_mask_whole_group_at_cap
uniform_success_group_behavior=clear_advantages_returns_response_mask_and_skip_optimizer
checkpoint_policy=one_temporary_recovery_checkpoint_at_actual_step5
full_training_allowed=false_pending_main_thread_review
EOF

export LLIN_CANARY_TARGET_OPTIMIZER_STEPS="${TARGET_ACTUAL_OPTIMIZER_STEPS}"
export PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}"
export PI_AGENT_RUN_TAG="${RUN_NAME}"

MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${MAX_NOMINAL_BATCHES}" \
TOTAL_ROLLOUT_GROUPS="${MAX_NOMINAL_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES=16 \
PREWARM_GROUPS=0 \
STALENESS_THRESHOLD=0 \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=90112 \
MAX_CONTEXT_TOKENS=94208 \
MAX_QUEUE_TOKENS="$((GROUPS_PER_STEP * RESPONSES_PER_GROUP * 94208))" \
MAX_ASSISTANT_TURNS=26 MAX_USER_TURNS=25 \
AGENT_TIMEOUT_SECONDS=1800 \
MAX_PARALLEL_TOOL_CALLS=4 MAX_TOOL_RESPONSE_CHARS=32768 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 ROLLOUT_MAX_SEQS=16 \
AGENT_WORKERS=12 CONCURRENT_SAMPLES_PER_REPLICA=6 \
WEIGHT_BUCKET_MB=2560 \
OPTIMIZER_CPU_OFFLOAD=false ENGINE_OPTIMIZER_OFFLOAD=false \
SAVE_FREQ="${TARGET_ACTUAL_OPTIMIZER_STEPS}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${SEALED_FILE}" \
  data.shuffle=false data.seed=20260822 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools_relaxed1800.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.actor.optim.lr=5e-8 \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score_tiered_query_cost_v1 \
  trainer.project_name=llin-qwen38-verl-grpo \
  trainer.val_before_train=true \
  trainer.test_freq="${TARGET_ACTUAL_OPTIMIZER_STEPS}" \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/sealed_validation" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/private/rollouts" \
  trainer.default_local_dir="${OUTPUT_DIR}/private_recovery/checkpoints" \
  trainer.save_freq="${TARGET_ACTUAL_OPTIMIZER_STEPS}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]'
