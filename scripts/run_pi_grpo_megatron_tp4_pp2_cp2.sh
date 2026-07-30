#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/data/pi_verified_smoke.parquet}"
RUN_NAME="${RUN_NAME:-pi-grpo-megatron-tp4-pp2-cp2-one-step-20260730}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"

TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-2}"
TRAIN_CP="${TRAIN_CP:-2}"
TRAIN_NPUS="${TRAIN_NPUS:-16}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"
ROLLOUT_NPUS="${ROLLOUT_NPUS:-8}"
SAVE_FREQ="${SAVE_FREQ:--1}"

if (( TRAIN_TP * TRAIN_PP * TRAIN_CP != TRAIN_NPUS )); then
  printf 'Invalid training topology: TP(%s) * PP(%s) * CP(%s) != NPUs(%s)\n' \
    "${TRAIN_TP}" "${TRAIN_PP}" "${TRAIN_CP}" "${TRAIN_NPUS}" >&2
  exit 2
fi

if (( ROLLOUT_NPUS % ROLLOUT_TP != 0 )); then
  printf 'Invalid rollout topology: NPUs(%s) is not divisible by TP(%s)\n' \
    "${ROLLOUT_NPUS}" "${ROLLOUT_TP}" >&2
  exit 2
fi

if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'Megatron-Bridge de93536e source not found: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
export LLIN_PIN_RAY_ROLES=1
export LLIN_TRAINER_RESOURCE=llin_trainer
export LLIN_ROLLOUT_RESOURCE=llin_rollout
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

python3 "${PROJECT_ROOT}/scripts/patch_verl_megatron_bridge_compat.py" \
  --target "${VERL_ROOT}/verl/models/mcore/bridge.py"

cd "${VERL_ROOT}"

python3 -m verl.experimental.one_step_off_policy.main_ppo \
  --config-path=config \
  --config-name=one_step_off_ppo_megatron_trainer.yaml \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=4 \
  data.dataloader_num_workers=4 \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs=False \
  data.truncation=error \
  actor_rollout_ref.actor.strategy=megatron \
  critic.strategy=megatron \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
  actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True \
  actor_rollout_ref.actor.checkpoint.strict=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.megatron.use_mbridge=True \
  actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
  actor_rollout_ref.ref.megatron.use_remove_padding=False \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
  actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.ref.megatron.dtype=bfloat16 \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.ref.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  +actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size="$((ROLLOUT_NPUS / ROLLOUT_TP))" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_sqlite_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=5120 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.project_name=llin-qwen36-verl-grpo \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.rollout_data_dir=null \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${TRAIN_NPUS}" \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node="${ROLLOUT_NPUS}" \
  "$@"
