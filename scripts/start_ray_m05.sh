#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
mkdir -p "${PROJECT_ROOT}/runs/ray/m05"

export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LLIN_PIN_RAY_ROLES=1
export HCCL_IF_IP=192.168.202.5
export HCCL_SOCKET_IFNAME=eno0
export HCCL_IF_BASE_PORT=60000
export HCCL_HOST_SOCKET_PORT_RANGE=60100-60163
export HCCL_NPU_SOCKET_PORT_RANGE=60200-60263

ray start \
  --head \
  --node-ip-address=192.168.202.5 \
  --port=26379 \
  --num-cpus=64 \
  --resources='{"llin_trainer": 1}' \
  --include-dashboard=false \
  --disable-usage-stats \
  --min-worker-port=27000 \
  --max-worker-port=27999 \
  --temp-dir="${PROJECT_ROOT}/runs/ray/m05"
