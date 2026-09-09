#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=/opt/llin-grounded-supply-sft-cpt-20260909-01
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
DATA="$ROOT/runs/supply-chain-task-training-data-20260909-01"
BOOK="$ROOT/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01"
BASE="$BOOK/hf_export_step_116"
SOURCE="$BOOK/checkpoints/global_step_116/model/dist_ckpt"
BRIDGE="$ROOT/reference/Megatron-Bridge-de93536e/src"
umask 077
[[ ! -e "$OUT" && -f "$SOURCE/.metadata" ]] || exit 2
[[ "$(cat "$ROOT/runs/supply-chain-task-baselines-20260909-01/status.txt")" == baselines_complete_training_not_started ]] || exit 4
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
[[ "$(df -B1 --output=avail /opt | tail -n 1 | tr -d ' ')" -ge 180000000000 ]] || exit 2
python3 -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]' "$DATA/train.parquet" 2f10b42c9a8ad0bde49b2c1887a6216727352c4871d1afdc5e84145e70dd9051
python3 -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]' "$DATA/dev.parquet" 54410c6d33db3cbbdcaa542a6a3a978f10d5acd4470dd8d132493629127c0e3a
mkdir "$OUT"
exec >"$OUT/pipeline.log" 2>&1
trap 'printf "failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$ROOT:$BRIDGE:$ROOT/runtime:/verl:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
unset ASCEND_RT_VISIBLE_DEVICES
printf 'training\n' > "$OUT/status.txt"
torchrun --standalone --nnodes=1 --nproc_per_node=16 --log-dir="$OUT/torchrun_logs" --redirects=3 --tee=0 \
 -m verl.trainer.sft_trainer \
 "data.train_files=$DATA/train.parquet" "data.val_files=$DATA/dev.parquet" \
 data.train_max_samples=-1 data.train_batch_size=8 data.micro_batch_size_per_gpu=1 \
 data.use_dynamic_bsz=false data.max_token_len_per_gpu=4096 data.max_length=4096 \
 data.pad_mode=no_padding data.truncation=error data.num_workers=0 \
 "data.custom_cls.path=$CODE/qwen36_mcq_answer_dataset.py" data.custom_cls.name=Qwen36MCQAnswerDataset \
 model=hf_model "model.path=$BASE" model.trust_remote_code=true model.use_remove_padding=false model.lora_rank=0 \
 model.mtp.enable=false model.mtp.enable_train=false model.mtp.enable_rollout=false \
 optim=megatron optim.lr=2e-7 optim.min_lr=2e-7 optim.lr_warmup_steps_ratio=0.01 \
 optim.weight_decay=0 'optim.betas=[0.9,0.999]' optim.clip_grad=1.0 optim.lr_decay_style=cosine \
 +optim.override_optimizer_config.adam_beta1=0.9 +optim.override_optimizer_config.adam_beta2=0.999 \
 +optim.override_optimizer_config.optimizer_cpu_offload=false \
 engine=megatron engine.tensor_model_parallel_size=4 engine.pipeline_model_parallel_size=2 engine.context_parallel_size=2 \
 engine.use_mbridge=true engine.vanilla_mbridge=false engine.use_megatron_fsdp=false engine.use_remove_padding=false \
 engine.param_offload=false engine.optimizer_offload=false engine.grad_offload=true engine.dtype=bfloat16 \
 engine.use_distributed_optimizer=true engine.use_dist_checkpointing=true "engine.dist_checkpointing_path=$SOURCE" \
 ++engine.override_transformer_config.attention_backend=auto ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
 ++engine.override_transformer_config.recompute_method=uniform ++engine.override_transformer_config.recompute_granularity=full \
 ++engine.override_transformer_config.recompute_num_layers=1 ++engine.override_transformer_config.use_flash_attn=true \
 ++engine.override_transformer_config.sequence_parallel=true 'checkpoint.load_contents=[]' 'checkpoint.save_contents=[model,extra]' \
 "trainer.default_local_dir=$OUT/checkpoints" trainer.project_name=llin-book-sft \
 trainer.experiment_name=grounded-mixed-tasks 'trainer.logger=["console"]' \
 trainer.total_epochs=1 trainer.total_training_steps=73 trainer.save_freq=73 trainer.test_freq=73 \
 trainer.resume_mode=disable trainer.max_ckpt_to_keep=1 trainer.nnodes=1 trainer.n_gpus_per_node=16
printf 'exporting\n' > "$OUT/status.txt"
python3 "$CODE/export_megatron_dist_to_hf.py" --actor-checkpoint "$OUT/checkpoints/global_step_73" \
 --base-model "$BASE" --output-dir "$OUT/hf_export" > "$OUT/export.log" 2>&1
printf 'training_export_complete_validation_pending\n' > "$OUT/status.txt"
