#!/usr/bin/env bash
set -euo pipefail

ray start \
  --address=192.168.202.5:26379 \
  --node-ip-address=192.168.202.4 \
  --num-cpus=64 \
  --resources='{"llin_rollout": 1}' \
  --disable-usage-stats \
  --min-worker-port=27000 \
  --max-worker-port=27999
