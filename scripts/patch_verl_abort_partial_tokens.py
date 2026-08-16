#!/usr/bin/env python3
"""Aggregate pre-abort vLLM token counts at the logical request layer."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_ABORT_PARTIAL_TOKENS_V1"
PREREQUISITE_MARKER = "LLIN_FASTEST_K_PHYSICAL_ABORT_V3"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_client(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    if PREREQUISITE_MARKER not in text:
        raise RuntimeError(
            f"{path} must first be patched by patch_verl_fastest_k_abort_retry.py"
        )

    text = _replace_once(
        text,
        '        """LLIN_FASTEST_K_PHYSICAL_ABORT_V3: retry until registered, aborted, or completed."""\n',
        '        """LLIN_FASTEST_K_PHYSICAL_ABORT_V3 / LLIN_ABORT_PARTIAL_TOKENS_V1."""\n',
        path,
    )
    text = _replace_once(
        text,
        '''\
                "physical_request_count": 0,
                "acknowledged_count": 0,
''',
        '''\
                "physical_request_count": 0,
                "partial_response_tokens": 0,
                "acknowledged_count": 0,
''',
        path,
    )
    text = _replace_once(
        text,
        '''\
        acknowledged_count = sum(isinstance(result, dict) for result in results)
        aborted_count = sum(
''',
        '''\
        acknowledged_count = sum(isinstance(result, dict) for result in results)
        partial_response_tokens = sum(
            max(0, int(result.get("partial_response_tokens", 0) or 0))
            for result in results
            if isinstance(result, dict)
        )
        aborted_count = sum(
''',
        path,
    )
    text = _replace_once(
        text,
        '''\
            "physical_request_count": len(active),
            "acknowledged_count": acknowledged_count,
''',
        '''\
            "physical_request_count": len(active),
            "partial_response_tokens": partial_response_tokens,
            "acknowledged_count": acknowledged_count,
''',
        path,
    )
    path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-server",
        default="/verl/verl/workers/rollout/llm_server.py",
    )
    args = parser.parse_args()
    print(f"{patch_client(Path(args.llm_server))}: {args.llm_server}")


if __name__ == "__main__":
    main()
