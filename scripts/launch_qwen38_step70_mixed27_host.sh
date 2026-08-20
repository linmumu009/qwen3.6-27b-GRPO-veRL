#!/usr/bin/env bash
set -euo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
POOL_DIR="${POOL_DIR:-${CONTAINER_PROJECT_ROOT}/runs/llin-qwen38-step70-mixed27-4x-20260820-01/data}"

RUN_NAME="${RUN_NAME:-llin-qwen38-step70-mixed27-4x-banded-v2-20260820-01}" \
POOL_DIR="${POOL_DIR}" \
CANONICAL_FILE="${POOL_DIR}/train27.sensitive.parquet" \
TRAIN_FILE="${POOL_DIR}/train27x4.sensitive.parquet" \
SEALED_FILE="${POOL_DIR}/sealed6.sensitive.parquet" \
SAFE_SUMMARY="${POOL_DIR}/train27x4.safe.json" \
ASSEMBLER_SCRIPT=assemble_qwen38_step70_mixed27.py \
TRAINING_SCRIPT=run_pi_qwen38_step70_mixed27_4x_banded_v2.sh \
MODEL_PATH="${CONTAINER_PROJECT_ROOT}/exports/llin-qwen38-grpo-step70-hf-20260819-02" \
MODEL_EXPORT_POLICY_STEP=70 \
EXPECTED_CHECKPOINT_STEP=54 \
exec bash "${HOST_PROJECT_ROOT}/scripts/launch_qwen38_train70_host.sh"
