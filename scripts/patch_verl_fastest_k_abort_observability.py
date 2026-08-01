#!/usr/bin/env python3
"""Upgrade Fastest-K cancellation logs with physical-request evidence."""

from __future__ import annotations

import argparse
from pathlib import Path


CLIENT_MARKER = "LLIN_FASTEST_K_PHYSICAL_ABORT_V2"
AGENT_MARKER = "LLIN_FASTEST_K_QUORUM_V2"
CLIENT_MARKER_V3 = "LLIN_FASTEST_K_PHYSICAL_ABORT_V3"
AGENT_MARKER_V3 = "LLIN_FASTEST_K_QUORUM_V3"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_client(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if CLIENT_MARKER in text or CLIENT_MARKER_V3 in text:
        return "already-patched"

    old = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = False) -> dict[str, Any]:
        """Abort every active physical request for one logical trajectory."""
        async with self._llin_active_requests_lock:
            active = list(self._llin_active_requests.get(str(request_id), {}).items())
        if not active:
            return {"logical_request_id": str(request_id), "aborted_count": 0, "results": []}

        results = await asyncio.gather(
            *[
                server.abort_request.remote(physical_id, reset_prefix_cache)
                for physical_id, (_server_id, server) in active
            ],
            return_exceptions=True,
        )
        aborted_count = sum(
            isinstance(result, dict) and bool(result.get("aborted"))
            for result in results
        )
        return {
            "logical_request_id": str(request_id),
            "aborted_count": aborted_count,
            "physical_request_count": len(active),
            "reset_prefix_cache": bool(reset_prefix_cache),
            "results": results,
        }
'''
    new = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = False) -> dict[str, Any]:
        """LLIN_FASTEST_K_PHYSICAL_ABORT_V2: abort and report acknowledgement evidence."""
        async with self._llin_active_requests_lock:
            active = list(self._llin_active_requests.get(str(request_id), {}).items())
        if not active:
            return {
                "logical_request_id": str(request_id),
                "aborted_count": 0,
                "physical_request_count": 0,
                "acknowledged_count": 0,
                "not_found_count": 0,
                "error_count": 0,
                "reset_prefix_cache": bool(reset_prefix_cache),
                "results": [],
            }

        results = await asyncio.gather(
            *[
                server.abort_request.remote(physical_id, reset_prefix_cache)
                for physical_id, (_server_id, server) in active
            ],
            return_exceptions=True,
        )
        acknowledged_count = sum(isinstance(result, dict) for result in results)
        aborted_count = sum(
            isinstance(result, dict) and bool(result.get("aborted")) for result in results
        )
        not_found_count = sum(
            isinstance(result, dict)
            and not bool(result.get("aborted"))
            and "not found" in str(result.get("error", "")).lower()
            for result in results
        )
        error_count = sum(isinstance(result, BaseException) for result in results) + sum(
            isinstance(result, dict)
            and not bool(result.get("aborted"))
            and "not found" not in str(result.get("error", "")).lower()
            for result in results
        )
        return {
            "logical_request_id": str(request_id),
            "aborted_count": aborted_count,
            "physical_request_count": len(active),
            "acknowledged_count": acknowledged_count,
            "not_found_count": not_found_count,
            "error_count": error_count,
            "reset_prefix_cache": bool(reset_prefix_cache),
            "results": results,
        }
'''
    path.write_text(_replace_once(text, old, new, path), encoding="utf-8")
    return "patched"


def patch_agent_loop(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if AGENT_MARKER in text or AGENT_MARKER_V3 in text:
        return "already-patched"

    old = '''\
            physically_aborted = sum(
                result.get("aborted_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            print(
                "[LLIN_FASTEST_K] "
                f"candidates={len(tasks)} selected={len(outputs)} "
                f"discarded={len(tasks) - len(outputs)} "
                f"completed_discarded={completed_but_discarded} "
                f"physical_aborts={physically_aborted} "
                f"quorum_s={time.monotonic() - started_at:.6f} "
                "reset_prefix_cache=False"
            )
'''
    new = '''\
            # LLIN_FASTEST_K_QUORUM_V2: distinguish inactive candidates from
            # active physical requests and failed abort acknowledgements.
            physically_aborted = sum(
                result.get("aborted_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            active_requests = sum(
                result.get("physical_request_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            abort_acks = sum(
                result.get("acknowledged_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            abort_not_active = sum(
                int(result.get("physical_request_count", 0) == 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            abort_failures = sum(
                result.get("not_found_count", 0) + result.get("error_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            ) + sum(not isinstance(result, dict) for result in abort_results)
            print(
                "[LLIN_FASTEST_K] "
                f"candidates={len(tasks)} selected={len(outputs)} "
                f"discarded={len(tasks) - len(outputs)} "
                f"completed_discarded={completed_but_discarded} "
                f"physical_aborts={physically_aborted} "
                f"quorum_s={time.monotonic() - started_at:.6f} "
                f"active_requests={active_requests} abort_acks={abort_acks} "
                f"abort_not_active={abort_not_active} abort_failures={abort_failures} "
                "reset_prefix_cache=False"
            )
'''
    path.write_text(_replace_once(text, old, new, path), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-loop",
        default="/verl/verl/experimental/agent_loop/agent_loop.py",
    )
    parser.add_argument(
        "--llm-server",
        default="/verl/verl/workers/rollout/llm_server.py",
    )
    args = parser.parse_args()
    print(f"{patch_agent_loop(Path(args.agent_loop))}: {args.agent_loop}")
    print(f"{patch_client(Path(args.llm_server))}: {args.llm_server}")


if __name__ == "__main__":
    main()
