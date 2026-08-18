#!/usr/bin/env python3
"""Mask timed-out agent outputs that have no vLLM rollout log-prob tensor."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_NONE_ROLLOUT_LOGPROBS_ZERO_LOSS_V2"
OLD_MARKER = "LLIN_NONE_ROLLOUT_LOGPROBS_ZERO_LOSS_V1"


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    if OLD_MARKER in text:
        upgraded = text.replace(OLD_MARKER, MARKER, 1).replace(
            "if any(input.response_logprobs is not None for input in inputs):",
            "if inputs:",
            1,
        )
        if upgraded == text:
            raise RuntimeError(f"could not upgrade missing-logprob patch: {path}")
        path.write_text(upgraded, encoding="utf-8")
        return "upgraded"
    old = """\
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
"""
    new = """\
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
    if old not in text:
        raise RuntimeError(f"expected agent-loop postprocess anchor not found: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-loop", default="/verl/verl/experimental/agent_loop/agent_loop.py")
    args = parser.parse_args()
    print(f"{patch(Path(args.agent_loop))}: {args.agent_loop}")


if __name__ == "__main__":
    main()
