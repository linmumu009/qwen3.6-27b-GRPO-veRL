#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
mkdir -p "${PROJECT_ROOT}/runs/ray/m05"

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
