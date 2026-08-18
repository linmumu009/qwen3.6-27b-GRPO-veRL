#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
HOST_LABEL="${HOST_LABEL:?HOST_LABEL is required (m00, m05, or m06)}"
EXPECTED_APPROVED="${EXPECTED_APPROVED:?EXPECTED_APPROVED is required}"
RUN_ID="${RUN_ID:-llin-banded-v2-strict-replay-20260818-01}"
RERUN_STEM="${RERUN_STEM:-llin-qwen38-v15-v20-v21-rerun-${HOST_LABEL}-20260817-01}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-${PROJECT_ROOT}/scripts/replay_strict_table_reward_gate.py}"

approved="${PROJECT_ROOT}/runs/llin-qwen38-grpo-audit-70-20260818-01/applied/${HOST_LABEL}/semantic_approved_candidates.sensitive.parquet"
state_root="${PROJECT_ROOT}/runs/${RERUN_STEM}-state"
output_root="${PROJECT_ROOT}/runs/${RUN_ID}/${HOST_LABEL}"
mkdir -p "${output_root}"

args=(
  python3 "${REPLAY_SCRIPT}"
  --approved "${approved}"
  --output-safe-json "${output_root}/safe_summary.json"
  --output-qualified-parquet "${output_root}/strict_qualified_candidates.sensitive.parquet"
  --expected-approved "${EXPECTED_APPROVED}"
  --host-label "${HOST_LABEL}"
)

for version in v15 v20 v21; do
  for wave in 2 4 6; do
    case "${wave}" in
      2) dataset="${state_root}/${version}/initial.sensitive.parquet" ;;
      4) dataset="${state_root}/${version}/unresolved2.sensitive.parquet" ;;
      6) dataset="${state_root}/${version}/unresolved4.sensitive.parquet" ;;
    esac
    run_dir="${PROJECT_ROOT}/runs/${RERUN_STEM}-${version}-wave${wave}"
    [[ -f "${dataset}" ]] || { echo "missing dataset: ${dataset}" >&2; exit 3; }
    [[ -d "${run_dir}/shards" ]] || { echo "missing shards: ${run_dir}/shards" >&2; exit 3; }
    args+=(--wave "${version}-wave${wave}" "${dataset}" "${run_dir}")
  done
done

exec "${args[@]}"
