#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
HOST_LABEL="${HOST_LABEL:?HOST_LABEL is required}"
EXPECTED_APPROVED="${EXPECTED_APPROVED:?EXPECTED_APPROVED is required}"
RERUN_STEM="${RERUN_STEM:-llin-qwen38-v15-v20-v21-rerun-${HOST_LABEL}-20260817-01}"
STATE_ROOT="${PROJECT_ROOT}/runs/${RERUN_STEM}-state"
APPROVED="${PROJECT_ROOT}/runs/llin-qwen38-grpo-audit-70-20260818-01/applied/${HOST_LABEL}/semantic_approved_candidates.sensitive.parquet"

args=(
  python3 "${PROJECT_ROOT}/scripts/summarize_qwen38_candidate_lengths.py"
  --approved "${APPROVED}"
  --expected-tasks "${EXPECTED_APPROVED}"
  --tokenizer /models/Qwen3.8-27B
)
for version in v15 v20 v21; do
  for wave in 2 4 6; do
    case "${wave}" in
      2) dataset="${STATE_ROOT}/${version}/initial.sensitive.parquet" ;;
      4) dataset="${STATE_ROOT}/${version}/unresolved2.sensitive.parquet" ;;
      6) dataset="${STATE_ROOT}/${version}/unresolved4.sensitive.parquet" ;;
    esac
    run_dir="${PROJECT_ROOT}/runs/${RERUN_STEM}-${version}-wave${wave}"
    [[ -f "${dataset}" ]] || { printf 'missing dataset: %s\n' "${dataset}" >&2; exit 3; }
    [[ -d "${run_dir}/shards" ]] || { printf 'missing shard directory: %s\n' "${run_dir}" >&2; exit 3; }
    args+=(--wave "${dataset}" "${run_dir}")
  done
done
exec "${args[@]}"
