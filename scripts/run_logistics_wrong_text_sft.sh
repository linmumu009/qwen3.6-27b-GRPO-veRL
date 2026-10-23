#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
RUN=${WRONG_TEXT_RUN:?}
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
BASE=$ROOT/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
BRIDGE=$ROOT/reference/Megatron-Bridge-de93536e/src
CASES=$ROOT/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl
umask 077
[[ -d $RUN && ! -e $RUN/pipeline.log && -f $BASE/model.safetensors.index.json ]] || exit 2
[[ $(df -B1 --output=avail "$RUN" | tail -n 1 | tr -d ' ') -ge 240000000000 ]] || exit 2
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
exec > "$RUN/pipeline.log" 2>&1
trap 'printf "failed_at_line_%s\n" "$LINENO" > "$RUN/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$(dirname -- "$CODE"):$ROOT:$BRIDGE:$ROOT/runtime:/verl:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export HCCL_HOST_SOCKET_PORT_RANGE=41000-41199 HCCL_NPU_SOCKET_PORT_RANGE=42000-42199
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$RUN/private" "$RUN/safe" "$RUN/hf" "$RUN/torchrun_logs"
python3 -c 'import importlib.util,pathlib,sys; p=pathlib.Path(importlib.util.find_spec("megatron.bridge").origin).resolve(); assert p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()),str(p)' "$BRIDGE"
printf 'checking_concurrent_communication\n' > "$RUN/status.txt"
torchrun --standalone --nnodes=1 --nproc_per_node=16 "$CODE/check_logistics_eval_ports.py" > "$RUN/ports.log" 2>&1
printf 'preparing_data\n' > "$RUN/status.txt"
python3 "$CODE/prepare_logistics_wrong_text_sft.py" --source "$CASES" \
  --baseline "$RUN/inputs/step120_opensource_nothinking.jsonl" \
  --native "$RUN/inputs/base_local_qwen36_27b_nothinking.jsonl" --model "$BASE" --output "$RUN/data"
read -r BATCH STEPS < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["batch_size"],d["total_steps"])' "$RUN/data/manifest.safe.json")
printf 'evaluating_step_0\n' > "$RUN/status.txt"
bash "$CODE/evaluate_logistics_wrong_text_checkpoint.sh" 0 > "$RUN/eval_step_0.log" 2>&1
printf 'training\n' > "$RUN/status.txt"
torchrun --standalone --nnodes=1 --nproc_per_node=16 --log-dir="$RUN/torchrun_logs" --redirects=3 --tee=0 \
 "$CODE/train_logistics_wrong_text_sft.py" \
 "data.train_files=$RUN/data/train.parquet" 'data.val_files=[]' \
 data.train_max_samples=-1 "data.train_batch_size=$BATCH" data.micro_batch_size_per_gpu=1 \
 data.use_dynamic_bsz=false data.max_token_len_per_gpu=4096 data.max_length=4096 \
 data.pad_mode=no_padding data.truncation=error data.num_workers=0 \
 "data.custom_cls.path=$CODE/qwen36_mcq_answer_dataset.py" data.custom_cls.name=Qwen36MCQAnswerDataset \
 model=hf_model "model.path=$BASE" model.trust_remote_code=true model.use_remove_padding=false model.lora_rank=0 \
 model.mtp.enable=false model.mtp.enable_train=false model.mtp.enable_rollout=false \
 optim=megatron optim.lr=2e-6 optim.min_lr=2e-6 optim.lr_warmup_steps_ratio=0.01 \
 optim.weight_decay=0 'optim.betas=[0.9,0.999]' optim.clip_grad=1.0 optim.lr_decay_style=cosine \
 +optim.override_optimizer_config.adam_beta1=0.9 +optim.override_optimizer_config.adam_beta2=0.999 \
 +optim.override_optimizer_config.optimizer_cpu_offload=false \
 engine=megatron engine.seed=1024 engine.tensor_model_parallel_size=4 engine.pipeline_model_parallel_size=2 engine.context_parallel_size=2 \
 engine.use_mbridge=true engine.vanilla_mbridge=false engine.use_megatron_fsdp=false engine.use_remove_padding=false \
 engine.param_offload=false engine.optimizer_offload=false engine.grad_offload=true engine.dtype=bfloat16 \
 engine.use_distributed_optimizer=true engine.use_dist_checkpointing=true engine.dist_checkpointing_path=null \
 ++engine.override_transformer_config.attention_backend=auto ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
 ++engine.override_transformer_config.recompute_method=uniform ++engine.override_transformer_config.recompute_granularity=full \
 ++engine.override_transformer_config.recompute_num_layers=1 ++engine.override_transformer_config.use_flash_attn=true \
 ++engine.override_transformer_config.sequence_parallel=true 'checkpoint.load_contents=[]' 'checkpoint.save_contents=[model,extra]' \
 "trainer.default_local_dir=$RUN/checkpoints" trainer.project_name=llin-logistics-wrong-text \
 trainer.experiment_name=wrong-union-full-text-one-epoch 'trainer.logger=["console"]' \
 trainer.total_epochs=1 "trainer.total_training_steps=$STEPS" trainer.save_freq=50 trainer.test_freq=-1 \
 trainer.seed=1024 trainer.resume_mode=disable trainer.max_ckpt_to_keep=1 trainer.nnodes=1 trainer.n_gpus_per_node=16
python3 - "$RUN" <<'PY'
import json, pathlib, shutil, sys
run=pathlib.Path(sys.argv[1]).resolve()
manifest=json.loads((run/'data/manifest.safe.json').read_text())
selection=json.loads((run/'selection.safe.json').read_text())
assert [r['step'] for r in selection['evaluations']]==[0]+manifest['evaluation_steps']
assert (run/'best_model').resolve().is_dir()
# Only this run's temporary distributed checkpoints are removed after all evaluations succeed.
checkpoints=run/'checkpoints'
assert checkpoints.resolve().parent==run and not checkpoints.is_symlink()
shutil.rmtree(checkpoints)
(run/'status.txt').write_text('complete\n')
print(json.dumps(selection['best'],indent=2))
PY
