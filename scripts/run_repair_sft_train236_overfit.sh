#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_train236_20260811}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/repair_sft_train.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
TOKENIZATION_GATE="${TOKENIZATION_GATE:-${DATA_DIR}/tokenization_gate.json}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-train236-overfit-step120-20260811-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TOTAL_STEPS="${TOTAL_STEPS:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid SFT topology: TP(%s) * PP(%s) * CP(%s) != NPROC(%s)\n' \
    "${TP}" "${PP}" "${CP}" "${NPROC}" >&2
  exit 2
fi
for path in \
  "${MODEL_PATH}/config.json" \
  "${SOURCE_MODEL_DIST_CKPT}/.metadata" \
  "${TRAIN_FILE}" \
  "${DATA_CONTRACT}" \
  "${TOKENIZATION_GATE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required repair SFT input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

python3 - "${DATA_CONTRACT}" "${TOKENIZATION_GATE}" "${TRAIN_BATCH_SIZE}" "${MAX_LENGTH}" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
token_gate = json.load(open(sys.argv[2], encoding="utf-8"))
batch_size = int(sys.argv[3])
max_length = int(sys.argv[4])
errors = []
if contract.get("contract") != "train236-repair-sft-dataset-v1":
    errors.append("unexpected data contract")
if contract.get("rows") != batch_size:
    errors.append("train batch must contain every repair row exactly once")
if contract.get("heldout_overlap") != 0:
    errors.append("repair rows overlap val/test")
for key in (
    "all_rows_approved",
    "all_rows_current_task_definition",
    "all_sql_read_only_executable_nonempty",
    "all_expected_values_match_sql",
):
    if contract.get(key) is not True:
        errors.append(f"data contract gate failed: {key}")
if token_gate.get("rows") != batch_size:
    errors.append("tokenization row count mismatch")
if token_gate.get("all_rows_have_assistant_loss") is not True:
    errors.append("one or more rows have no assistant loss")
if token_gate.get("all_rows_mask_non_assistant_context") is not True:
    errors.append("one or more rows leak context into the loss")
if max(sample["total_tokens"] for sample in token_gate.get("samples", [])) > max_length:
    errors.append("one or more rows exceed MAX_LENGTH")
if errors:
    raise SystemExit("; ".join(errors))
PY

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=train236_repair_sft_overfit_gate
promotion_allowed=false
source_checkpoint=${SOURCE_CHECKPOINT}
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
checkpoint_initialization=model_only_dist_ckpt
optimizer_state=fresh_cpu_offloaded_adam
dataloader_state=fresh
train_rows=16
train_split=train236_only
heldout_overlap=0
train_batch_size=${TRAIN_BATCH_SIZE}
micro_batch_size_per_gpu=1
total_training_steps=${TOTAL_STEPS}
total_epochs=${TOTAL_EPOCHS}
learning_rate=${LEARNING_RATE}
max_length=${MAX_LENGTH}
save_policy=final_model_and_extra_only
optimizer_checkpoint_saved=false
intermediate_validation=false
topology=tp${TP}_pp${PP}_cp${CP}
replay_gate=at_least_14_of_16_exact_boss_reward_success
heldout_promotion_gate=separate_48_to_64_repair_canary
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
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
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
  'checkpoint.save_contents=[model,extra]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  trainer.project_name=llin-repair-sft \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  "trainer.total_epochs=${TOTAL_EPOCHS}" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "$@"
