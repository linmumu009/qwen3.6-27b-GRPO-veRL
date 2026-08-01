#!/usr/bin/env python3
"""Close the registration race between Fastest-K cancellation and vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path


CLIENT_MARKER = "LLIN_FASTEST_K_PHYSICAL_ABORT_V3"
AGENT_MARKER = "LLIN_FASTEST_K_QUORUM_V3"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_client(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if CLIENT_MARKER in text:
        return "already-patched"

    old = '''\
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
    new = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = False) -> dict[str, Any]:
        """LLIN_FASTEST_K_PHYSICAL_ABORT_V3: retry until registered, aborted, or completed."""
        async with self._llin_active_requests_lock:
            active = list(self._llin_active_requests.get(str(request_id), {}).items())
        if not active:
            return {
                "logical_request_id": str(request_id),
                "aborted_count": 0,
                "physical_request_count": 0,
                "acknowledged_count": 0,
                "completed_before_abort_count": 0,
                "retry_exhausted_count": 0,
                "error_count": 0,
                "reset_prefix_cache": bool(reset_prefix_cache),
                "results": [],
            }

        async def abort_one(physical_id: str, server: Any) -> dict[str, Any]:
            last_result: dict[str, Any] | None = None
            for attempt in range(1, 21):
                result = await server.abort_request.remote(physical_id, reset_prefix_cache)
                last_result = result if isinstance(result, dict) else {"error": repr(result)}
                if isinstance(result, dict) and bool(result.get("aborted")):
                    return {**result, "attempts": attempt}
                async with self._llin_active_requests_lock:
                    still_active = physical_id in self._llin_active_requests.get(str(request_id), {})
                if not still_active:
                    return {
                        **last_result,
                        "completed_before_abort": True,
                        "attempts": attempt,
                    }
                if attempt < 20:
                    await asyncio.sleep(0.05)
            return {
                **(last_result or {}),
                "retry_exhausted": True,
                "attempts": 20,
            }

        results = await asyncio.gather(
            *[
                abort_one(physical_id, server)
                for physical_id, (_server_id, server) in active
            ],
            return_exceptions=True,
        )
        acknowledged_count = sum(isinstance(result, dict) for result in results)
        aborted_count = sum(
            isinstance(result, dict) and bool(result.get("aborted")) for result in results
        )
        completed_before_abort_count = sum(
            isinstance(result, dict) and bool(result.get("completed_before_abort"))
            for result in results
        )
        retry_exhausted_count = sum(
            isinstance(result, dict) and bool(result.get("retry_exhausted")) for result in results
        )
        error_count = sum(isinstance(result, BaseException) for result in results) + sum(
            isinstance(result, dict)
            and not bool(result.get("aborted"))
            and not bool(result.get("completed_before_abort"))
            and not bool(result.get("retry_exhausted"))
            for result in results
        )
        return {
            "logical_request_id": str(request_id),
            "aborted_count": aborted_count,
            "physical_request_count": len(active),
            "acknowledged_count": acknowledged_count,
            "completed_before_abort_count": completed_before_abort_count,
            "retry_exhausted_count": retry_exhausted_count,
            "error_count": error_count,
            "reset_prefix_cache": bool(reset_prefix_cache),
            "results": results,
        }
'''
    path.write_text(_replace_once(text, old, new, path), encoding="utf-8")
    return "patched"


def patch_agent_loop(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if AGENT_MARKER in text:
        return "already-patched"

    old = '''\
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
    new = '''\
            # LLIN_FASTEST_K_QUORUM_V3: every active candidate must be aborted,
            # confirmed complete, or reported as retry-exhausted before cancel.
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
            abort_completed = sum(
                result.get("completed_before_abort_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            abort_retry_exhausted = sum(
                result.get("retry_exhausted_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            abort_failures = sum(
                result.get("error_count", 0)
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
                f"abort_not_active={abort_not_active} abort_completed={abort_completed} "
                f"abort_retry_exhausted={abort_retry_exhausted} abort_failures={abort_failures} "
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
