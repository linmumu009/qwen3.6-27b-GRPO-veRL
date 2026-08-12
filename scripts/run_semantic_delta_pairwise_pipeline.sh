#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-semantic-delta-pairwise-step120-1step-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/semantic_delta_margin_gate_20260812}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${RUN_NAME}-baseline-forward}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}}"
POST_RUN_NAME="${POST_RUN_NAME:-${RUN_NAME}-post-forward}"
POST_OUTPUT_DIR="${POST_OUTPUT_DIR:-${PROJECT_ROOT}/runs/${POST_RUN_NAME}}"
DATA_CONTRACT="${DATA_DIR}/contract.json"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
RUN_NAME="${BASELINE_RUN_NAME}" \
OUTPUT_DIR="${BASELINE_OUTPUT_DIR}" \
MODEL_LABEL=step120_pairwise_pipeline_baseline \
bash "${PROJECT_ROOT}/scripts/run_semantic_delta_margin_gate.sh"

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_DIR="${DATA_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_semantic_delta_pairwise_canary.sh"

POST_MODEL_DIST_CKPT="${OUTPUT_DIR}/checkpoints/global_step_1/model/dist_ckpt"
if [[ ! -f "${POST_MODEL_DIST_CKPT}/.metadata" ]]; then
  printf 'pairwise step1 model checkpoint is incomplete\n' >&2
  exit 2
fi
if find "${OUTPUT_DIR}/checkpoints/global_step_1" -type f -path '*optimizer*' -print -quit | grep -q .; then
  printf 'pairwise canary unexpectedly saved optimizer state\n' >&2
  exit 2
fi

RUN_NAME="${POST_RUN_NAME}" \
OUTPUT_DIR="${POST_OUTPUT_DIR}" \
MODEL_DIST_CKPT="${POST_MODEL_DIST_CKPT}" \
MODEL_LABEL=pairwise_step1_semantic_delta_margin \
bash "${PROJECT_ROOT}/scripts/run_semantic_delta_margin_gate.sh"

python3 -m scripts.compare_semantic_delta_margin_canary \
  --baseline "${BASELINE_OUTPUT_DIR}/semantic_delta_margin_result.json" \
  --post "${POST_OUTPUT_DIR}/semantic_delta_margin_result.json" \
  --output "${OUTPUT_DIR}/comparison.json"
