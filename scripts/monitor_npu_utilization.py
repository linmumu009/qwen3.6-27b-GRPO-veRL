#!/usr/bin/env python3
"""Sample all Ascend chips into a CSV until an experiment exit file appears."""

from __future__ import annotations

import argparse
import csv
import re
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "timestamp",
    "host",
    "role",
    "completed_step",
    "card",
    "chip",
    "aicore_pct",
    "npu_util_pct",
    "hbm_usage_pct",
    "hbm_bandwidth_pct",
)
STEP_RE = re.compile(r"step:(\d+)\s+-\s+training/global_step")
STOP = False


def parse_usage(text: str) -> list[dict[str, int]]:
    """Parse the two chip records returned for one ``npu-smi`` card."""
    records: list[dict[str, int]] = []
    current: dict[str, int] = {}
    key_map = {
        "Aicore Usage Rate(%)": "aicore_pct",
        "NPU Utilization(%)": "npu_util_pct",
        "HBM Usage Rate(%)": "hbm_usage_pct",
        "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_pct",
    }
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = (part.strip() for part in raw_line.split(":", 1))
        if key in key_map:
            current[key_map[key]] = int(value)
        elif key == "Chip ID":
            current["chip"] = int(value)
            records.append(current)
            current = {}
    return records


class StepTracker:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.offset = 0
        self.completed_step = 0

    def update(self) -> int:
        if self.path is None or not self.path.exists():
            return self.completed_step
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        for match in STEP_RE.finditer(chunk):
            self.completed_step = max(self.completed_step, int(match.group(1)))
        return self.completed_step


def sample_card(card: int) -> list[dict[str, int]]:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "usages", "-i", str(card)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = parse_usage(result.stdout)
    for record in rows:
        record["card"] = card
    return rows


def sample(cards: range) -> list[dict[str, int]]:
    cards_list = list(cards)
    rows: list[dict[str, int]] = []
    with ThreadPoolExecutor(max_workers=len(cards_list)) as executor:
        card_rows = executor.map(sample_card, cards_list)
        for records in card_rows:
            rows.extend(records)
    return rows


def stop_handler(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--until-file", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--first-card", type=int, default=0)
    parser.add_argument("--num-cards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    tracker = StepTracker(args.driver_log)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        while not STOP and not args.until_file.exists():
            started = time.monotonic()
            timestamp = datetime.now(timezone.utc).isoformat()
            completed_step = tracker.update()
            for row in sample(range(args.first_card, args.first_card + args.num_cards)):
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "host": socket.gethostname(),
                        "role": args.role,
                        "completed_step": completed_step,
                        **row,
                    }
                )
            output.flush()
            delay = args.interval - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
