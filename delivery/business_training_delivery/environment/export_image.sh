#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=sha256:cdc98ccb6b739429aa81f3dc6e4bca9543d1e99202cd7e8686055fb5e9fae7b7
DEST="$ROOT/environment/ascend-verl-base.tar"
[[ ! -e "$DEST" && ! -e "$DEST.partial" ]] || { echo '镜像归档已存在，请先核查。'; exit 2; }
docker image inspect "$IMAGE" > /dev/null
docker save "$IMAGE" -o "$DEST.partial"
mv "$DEST.partial" "$DEST"
(cd "$(dirname "$DEST")" && sha256sum "$(basename "$DEST")" > "$(basename "$DEST").sha256")
echo "迁移时执行：docker load -i environment/ascend-verl-base.tar"
