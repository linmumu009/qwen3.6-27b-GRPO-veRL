#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-pi-dense-correctness-step100-to-step120-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

# Isolate reward shaping first: topology, prompts/update, response count,
# context, rollout capacity, LR, checkpoint source, and dataset remain equal
# to the prior Step-100 continuation.  Only 30% dense correctness is enabled.
FINAL_POLICY_STEP=120 \
PI_DENSE_CORRECTNESS_WEIGHT=0.30 \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_pi_formal_step100_to_step200_12groups.sh" \
  reward.custom_reward_function.name=compute_score_dense30 \
  "$@"
