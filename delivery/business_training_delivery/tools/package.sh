#!/usr/bin/env bash
# Package inputs and runtime, excluding validation checkpoints and caches.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:?Usage: bash tools/package.sh /absolute/path/business_training_delivery.tar}
[[ "$DEST" = /* ]] || { echo '输出路径必须是绝对路径。'; exit 2; }
DEST=$(realpath -m -- "$DEST")
case "$DEST" in "$ROOT"/*) echo '归档输出不能放在交付目录内部。'; exit 2 ;; esac
[[ ! -e "$DEST" && ! -e "$DEST.partial" ]] || exit 2
[[ -d "$ROOT/model/qwen3.6-27b" && ! -L "$ROOT/model/qwen3.6-27b" ]] || exit 2
[[ -f "$ROOT/environment/ascend-verl-base.tar" ]] || exit 2
tar --exclude='*/__pycache__' --exclude='*.pyc' \
  --exclude='business_training_delivery/outputs/*/*' \
  --exclude='business_training_delivery/outputs/.*' \
  -cf "$DEST.partial" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
mv "$DEST.partial" "$DEST"
(cd "$(dirname "$DEST")" && sha256sum "$(basename "$DEST")" > "$(basename "$DEST").sha256")
echo "已生成：$DEST（CPT模型若仍为空，则保持预留）"
