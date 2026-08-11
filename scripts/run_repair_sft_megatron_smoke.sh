#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_smoke_20260811}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/repair_sft_smoke_train.parquet}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-megatron-smoke-20260811-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"

for path in "${MODEL_PATH}/config.json" "${SOURCE_MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required SFT smoke input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/check_repair_sft_dataset.py" \
  --train-file "${TRAIN_FILE}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --output "${OUTPUT_DIR}/tokenization_gate.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=throwaway_forward_backward_smoke
promotion_allowed=false
source_checkpoint=${SOURCE_CHECKPOINT}
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
checkpoint_resume=false
checkpoint_initialization=model_only_dist_ckpt
optimizer_state=fresh
dataloader_state=fresh
total_training_steps=1
save_contents=extra_only
topology=tp${TP}_pp${PP}_cp${CP}
EOF

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  "data.train_files=${TRAIN_FILE}" \
  data.val_files=null \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  "data.max_token_len_per_gpu=${MAX_LENGTH}" \
  "data.max_length=${MAX_LENGTH}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.ignore_input_ids_mismatch=false \
  "data.custom_cls.path=${PROJECT_ROOT}/scripts/qwen36_assistant_mask_sft_dataset.py" \
  data.custom_cls.name=Qwen36AssistantMaskSFTDataset \
  model=hf_model \
  "model.path=${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank=0 \
  optim=megatron \
  "optim.lr=${LEARNING_RATE}" \
  optim.lr_warmup_steps_ratio=0 \
  optim.weight_decay=0 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.lr_decay_style=constant \
  +optim.override_optimizer_config.optimizer_cpu_offload=true \
  +optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=true \
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.param_offload=false \
  engine.optimizer_offload=true \
  engine.grad_offload=true \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true \
  engine.use_dist_checkpointing=true \
  "engine.dist_checkpointing_path=${SOURCE_MODEL_DIST_CKPT}" \
  ++engine.override_transformer_config.attention_backend=auto \
  ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.recompute_method=uniform \
  ++engine.override_transformer_config.recompute_granularity=full \
  ++engine.override_transformer_config.recompute_num_layers=1 \
  ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true \
  'checkpoint.load_contents=[]' \
  'checkpoint.save_contents=[extra]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  trainer.project_name=llin-repair-sft \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "$@"
