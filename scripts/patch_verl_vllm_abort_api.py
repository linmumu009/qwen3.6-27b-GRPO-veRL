#!/usr/bin/env python3
"""Use vLLM's public external-request abort API on vLLM 0.18."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_VLLM_PUBLIC_ABORT_V4"


def patch_server(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"

    old = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}
'''
    new = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """LLIN_VLLM_PUBLIC_ABORT_V4: abort an external vLLM request ID."""
        try:
            # vLLM 0.18 request_states is keyed by internal IDs. The public
            # AsyncLLM.abort API resolves the external ID through
            # output_processor.external_req_ids before touching EngineCore.
            external_req_ids = getattr(self.engine.output_processor, "external_req_ids", {})
            registered = bool(external_req_ids.get(request_id))
            if not registered:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            await self.engine.abort(request_id)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request through public AsyncLLM API: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}
'''
    if old not in text:
        raise RuntimeError(f"expected abort_request anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py",
    )
    args = parser.parse_args()
    print(f"{patch_server(Path(args.target))}: {args.target}")


if __name__ == "__main__":
    main()
