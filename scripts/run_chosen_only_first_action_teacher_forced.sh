#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/chosen_only_schema_action_20260813}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/chosen_only_schema_action_calibration16.parquet}"
DATASET_CONTRACT="${DATASET_CONTRACT:-${DATA_DIR}/contract.json}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:?SOURCE_MODEL_DIST_CKPT is required}"
MODEL_LABEL="${MODEL_LABEL:-step120}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid topology: TP(%s) * PP(%s) * CP(%s) != NPROC(%s)\n' "${TP}" "${PP}" "${CP}" "${NPROC}" >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${SOURCE_MODEL_DIST_CKPT}/.metadata" "${DATA_FILE}" "${DATASET_CONTRACT}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required chosen-only diagnostic input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/check_chosen_only_schema_action_sft.py" \
  --data-file "${DATA_FILE}" \
  --dataset-contract "${DATASET_CONTRACT}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --output "${OUTPUT_DIR}/tokenization_gate.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=chosen_only_schema_conditioned_first_action_teacher_forced_baseline
model_label=${MODEL_LABEL}
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
data_file=${DATA_FILE}
task_count=16
forward_only=true
optimizer_initialized=false
checkpoint_saved=false
topology=tp${TP}_pp${PP}_cp${CP}
components=assistant,tool_turn,tool_structure,sql_shell
sql_token_rank=exact_vocab_parallel_rank
oracle_relevant_table_selection=true
deployment_ready=false
training_allowed=false
promotion_allowed=false
EOF

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m scripts.run_teacher_forced_component_diagnostic \
  "data.train_files=${DATA_FILE}" \
  "data.val_files=${DATA_FILE}" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  "data.max_token_len_per_gpu=${MAX_LENGTH}" \
  "data.max_length=${MAX_LENGTH}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.ignore_input_ids_mismatch=false \
  "data.custom_cls.path=${PROJECT_ROOT}/scripts/qwen36_first_action_diagnostic_dataset.py" \
  data.custom_cls.name=Qwen36FirstActionDiagnosticDataset \
  model=hf_model \
  "model.path=${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank=0 \
  optim=megatron \
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.forward_only=true \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.grad_offload=false \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true \
  engine.use_dist_checkpointing=true \
  "engine.dist_checkpointing_path=${SOURCE_MODEL_DIST_CKPT}" \
  ++engine.override_transformer_config.attention_backend=auto \
  ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true \
  'checkpoint.load_contents=[]' \
  'checkpoint.save_contents=[]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/unused_checkpoints" \
  trainer.project_name=llin-chosen-only-first-action-diagnostic \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "+diagnostic.output_path=${OUTPUT_DIR}/result.json" \
  "+diagnostic.model_label=${MODEL_LABEL}" \
  "$@"
