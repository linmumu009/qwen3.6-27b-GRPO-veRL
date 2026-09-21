#!/usr/bin/env bash
# Host entry point: a separate container for each call; no old Ray cluster reuse.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
KIND=${1:?training type required}
shift
case "$KIND" in SFT|trajory-SFT|GRPO|agentic-GRPO|CPT) ;; *) exit 2 ;; esac
IMAGE=${DELIVERY_IMAGE:-sha256:cdc98ccb6b739429aa81f3dc6e4bca9543d1e99202cd7e8686055fb5e9fae7b7}
MODEL_INPUT=${DELIVERY_MODEL_DIR:-"$ROOT/model/qwen3.6-27b"}
[[ "$MODEL_INPUT" = /* ]] || MODEL_INPUT="$ROOT/$MODEL_INPUT"
MODEL=$(readlink -f "$MODEL_INPUT")
test -f "$MODEL/model.safetensors.index.json"
test -f "$ROOT/frameworks/verl/verl/trainer/sft_trainer.py"
test -d "$ROOT/frameworks/Megatron-Bridge/src/megatron/bridge"
mkdir -p "$ROOT/outputs/$KIND"
exec 9>"$ROOT/outputs/.training.lock"
flock -n 9 || { echo '交付包已有任务正在运行。'; exit 3; }
# Cooperate with the historical machine05 CPT coordinator, if present.
OLD_LOCK=/data3/llin/qwen3.6-27b-verl-grpo/runs/.logistics-exam-cpt.lock
if [[ -d $(dirname "$OLD_LOCK") ]]; then
  exec 8>"$OLD_LOCK"
  flock -n 8 || { echo '历史训练任务持有设备锁，请稍后重试。'; exit 3; }
fi
if [[ " ${*:-} " != *' --check '* && " ${*:-} " != *' --compose '* ]]; then
  idle=$(npu-smi info | grep -c 'No running processes found' || true)
  [[ "$idle" == 8 ]] || { echo '设备正在使用，本次未启动训练。'; exit 3; }
  available=$(df -B1 --output=avail "$ROOT/outputs" | tail -n 1 | tr -d ' ')
  [[ "$available" -ge 90000000000 ]] || { echo '至少需要90GB可用空间保存检查点。'; exit 3; }
fi
STAMP=$(date +%Y%m%d-%H%M%S)-$$
OUT="$ROOT/outputs/$KIND/$STAMP"
mkdir "$OUT"
mkdir "$OUT/runtime_tmp"
DEVICES=()
for dev in /dev/davinci[0-9]* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ ! -e "$dev" ]] || DEVICES+=(--device "$dev")
done
set +e
docker run --rm --name "business-training-${KIND,,}-$STAMP" \
  --network host --ipc host --shm-size 32g --user 0 --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  "${DEVICES[@]}" \
  -v "$ROOT:/delivery" -v "$MODEL:/models/base:ro" -v "$OUT/runtime_tmp:/tmp" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -e PYTHONPATH=/delivery:/delivery/frameworks/verl:/delivery/frameworks/Megatron-Bridge/src \
  -e RAY_ADDRESS=local -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e HCCL_CONNECT_TIMEOUT=300 -e HCCL_EXEC_TIMEOUT=600 \
  -e TOKENIZERS_PARALLELISM=false -e HYDRA_FULL_ERROR=1 -e TRANSFORMERS_VERBOSITY=error \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e DELIVERY_TOOL_AUDIT="/delivery/outputs/$KIND/$STAMP/tool_calls.jsonl" \
  --workdir /delivery --entrypoint bash "$IMAGE" \
  /delivery/environment/inside.sh "$KIND" "/delivery/outputs/$KIND/$STAMP" "$@" \
  2>&1 | tee "$OUT/driver.log"
code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$code" > "$OUT/exit_code"
echo "结果目录：$OUT；退出码：$code"
exit "$code"
