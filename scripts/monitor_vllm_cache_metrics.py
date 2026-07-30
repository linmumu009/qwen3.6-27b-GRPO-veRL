#!/usr/bin/env python3
"""Sample vLLM prefix-cache counters from its dynamically assigned HTTP server."""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SERVER_RE = re.compile(r"LLMServerManager:\s*\['([^']+)'\]")
METRIC_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$"
)
KEEP_PARTS = (
    "prefix_cache",
    "cache_hit",
    "cache_query",
    "prompt_tokens",
    "generation_tokens",
)
STOP = False


def stop_handler(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def find_endpoint(driver_log: Path) -> str | None:
    if not driver_log.exists():
        return None
    text = driver_log.read_text(encoding="utf-8", errors="ignore")
    matches = SERVER_RE.findall(text)
    return matches[-1] if matches else None


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        if raw_line.startswith("#"):
            continue
        match = METRIC_RE.match(raw_line.strip())
        if not match:
            continue
        name, labels, value = match.groups()
        if any(part in name.lower() for part in KEEP_PARTS):
            metrics[f"{name}{labels or ''}"] = float(value)
    return metrics


def fetch_metrics(endpoint: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(f"http://{endpoint}/metrics", timeout=timeout) as response:
        return parse_metrics(response.read().decode("utf-8", errors="ignore"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    endpoint: str | None = None
    with args.output.open("a", encoding="utf-8") as output:
        while not STOP and not args.until_file.exists():
            started = time.monotonic()
            endpoint = endpoint or find_endpoint(args.driver_log)
            record: dict[str, object] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "metrics": {},
            }
            if endpoint:
                try:
                    record["metrics"] = fetch_metrics(endpoint, args.timeout)
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    endpoint = None
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            delay = args.interval - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
