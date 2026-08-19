#!/usr/bin/env python3
"""Zero-mask incomplete rollouts and normalize their missing version metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


MARKER = "LLIN_NONE_ROLLOUT_METADATA_ZERO_LOSS_V3"
DETACH_MARKER = "LLIN_NONE_PARAM_VERSIONS_NORMALIZED_V1"


ORIGINAL_AGENT_BLOCK = """\
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
"""


V2_AGENT_BLOCK = """\
        # LLIN_NONE_ROLLOUT_LOGPROBS_ZERO_LOSS_V2: an otherwise completed
        # timeout/abort can lack vLLM log-probs. Keep the group shape exact,
        # but give that trajectory a zero response mask so it contributes no
        # policy-gradient tokens. This is safer than crashing the whole queue
        # or inventing behavior-policy probabilities.
        missing_rollout_logprobs = [input.response_logprobs is None for input in inputs]
        response_mask = torch.cat(
            [
                torch.zeros_like(input.response_mask)
                if missing
                else input.response_mask
                for input, missing in zip(inputs, missing_rollout_logprobs)
            ],
            dim=0,
        )
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs:
            optional_outputs["rollout_log_probs"] = torch.cat(
                [
                    torch.zeros_like(input.response_ids, dtype=torch.float32)
                    if missing
                    else input.response_logprobs
                    for input, missing in zip(inputs, missing_rollout_logprobs)
                ],
                dim=0,
            )
"""


V3_AGENT_BLOCK = """\
        # LLIN_NONE_ROLLOUT_METADATA_ZERO_LOSS_V3: a timeout/cancellation can
        # return a shape-preserving trajectory without behavior log-probs or
        # model-version metadata. Such a trajectory must not contribute policy
        # gradient tokens even if one of those fields happens to be present.
        missing_rollout_metadata = [
            input.response_logprobs is None
            or input.extra_fields.get("min_global_steps") is None
            or input.extra_fields.get("max_global_steps") is None
            for input in inputs
        ]
        response_mask = torch.cat(
            [
                torch.zeros_like(input.response_mask)
                if missing
                else input.response_mask
                for input, missing in zip(inputs, missing_rollout_metadata)
            ],
            dim=0,
        )
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs:
            optional_outputs["rollout_log_probs"] = torch.cat(
                [
                    torch.zeros_like(input.response_ids, dtype=torch.float32)
                    if missing
                    else input.response_logprobs
                    for input, missing in zip(inputs, missing_rollout_metadata)
                ],
                dim=0,
            )
"""


ORIGINAL_DETACH_BLOCK = """\
    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
"""


PATCHED_DETACH_BLOCK = """\
    # LLIN_NONE_PARAM_VERSIONS_NORMALIZED_V1: cancelled/timeout trajectories
    # can have no model version because generation never returned a token.
    # They are zero-masked in agent_loop; normalize only the metadata used by
    # partial/staleness statistics so an invalid sample cannot crash the run.
    raw_param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    raw_param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    known_param_versions = [
        value
        for value in [*raw_param_version_start, *raw_param_version_end]
        if value is not None
    ]
    fallback_param_version = max(known_param_versions, default=0)
    normalized_param_version_start = []
    normalized_param_version_end = []
    for start, end in zip(raw_param_version_start, raw_param_version_end, strict=False):
        if start is None and end is None:
            start = end = fallback_param_version
        elif start is None:
            start = end
        elif end is None:
            end = start
        normalized_param_version_start.append(start)
        normalized_param_version_end.append(end)
    param_version_start = np.asarray(normalized_param_version_start, dtype=np.int64)
    param_version_end = np.asarray(normalized_param_version_end, dtype=np.int64)
    final_batch.non_tensor_batch["min_global_steps"] = param_version_start
    final_batch.non_tensor_batch["max_global_steps"] = param_version_end
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
"""


def normalize_param_versions(
    starts: Iterable[int | None], ends: Iterable[int | None]
) -> tuple[list[int], list[int]]:
    """Mirror the injected fail-closed version normalization for unit tests."""

    start_values = list(starts)
    end_values = list(ends)
    known = [value for value in [*start_values, *end_values] if value is not None]
    fallback = max(known, default=0)
    normalized_starts: list[int] = []
    normalized_ends: list[int] = []
    for start, end in zip(start_values, end_values, strict=False):
        if start is None and end is None:
            start = end = fallback
        elif start is None:
            start = end
        elif end is None:
            end = start
        normalized_starts.append(int(start))
        normalized_ends.append(int(end))
    return normalized_starts, normalized_ends


def patch_agent_loop(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    if V2_AGENT_BLOCK in text:
        source = V2_AGENT_BLOCK
        result = "upgraded"
    elif ORIGINAL_AGENT_BLOCK in text:
        source = ORIGINAL_AGENT_BLOCK
        result = "patched"
    else:
        raise RuntimeError(f"expected agent-loop postprocess anchor not found: {path}")
    path.write_text(text.replace(source, V3_AGENT_BLOCK, 1), encoding="utf-8")
    return result


def patch_detach_utils(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if DETACH_MARKER in text:
        return "already-patched"
    if ORIGINAL_DETACH_BLOCK not in text:
        raise RuntimeError(f"expected detach-utils version anchor not found: {path}")
    path.write_text(text.replace(ORIGINAL_DETACH_BLOCK, PATCHED_DETACH_BLOCK, 1), encoding="utf-8")
    return "patched"


def patch(path: Path) -> str:
    """Backward-compatible alias for the original agent-loop patch API."""

    return patch_agent_loop(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-loop", default="/verl/verl/experimental/agent_loop/agent_loop.py")
    parser.add_argument(
        "--detach-utils",
        default="/verl/verl/experimental/fully_async_policy/detach_utils.py",
    )
    args = parser.parse_args()
    print(f"{patch_agent_loop(Path(args.agent_loop))}: {args.agent_loop}")
    print(f"{patch_detach_utils(Path(args.detach_utils))}: {args.detach_utils}")


if __name__ == "__main__":
    main()
