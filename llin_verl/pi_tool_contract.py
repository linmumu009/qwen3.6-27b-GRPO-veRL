"""Pure helpers shared by PI tools, rewards, dataset QA, and local tests."""

from __future__ import annotations

import re


NETWORK_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:curl|wget|ssh|scp|nc|ncat|telnet|ftp|git\s+clone|"
    r"pip\s+install|apt(?:-get)?|dnf|yum)\b",
    re.IGNORECASE,
)
DESTRUCTIVE_PATTERN = re.compile(
    r"(?:\brm\s+-[^\n]*r[^\n]*\s+/(?:\s|$)|\bmkfs\b|\bdd\s+if=|"
    r"\b(?:shutdown|reboot|docker|podman|mount|umount|kill|killall|pkill)\b)",
    re.IGNORECASE,
)
ESCAPE_PATTERN = re.compile(
    r"(?:^|[\s'\"=])/(?:etc|proc|sys|dev|root|home|data|data3|models|"
    r"pi_sandbox|usr/local/Ascend)(?:/|\s|$)",
    re.IGNORECASE,
)
PYTHON_NETWORK_PATTERN = re.compile(
    r"(?:socket|urllib|requests|httpx|aiohttp|ftplib|smtplib)\s*[.(]",
    re.IGNORECASE,
)
ROOT_SCAN_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)(?:find|ls|du|tree)\s+(?:-[^\s]+\s+)*/(?:\s|$|\*)",
    re.IGNORECASE,
)
TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
ENVIRONMENT_PATTERN = re.compile(r"^sft/[A-Za-z0-9._-]+$")
SQLITE_COMMAND_PATTERN = re.compile(r"(?<![A-Za-z0-9_./-])sqlite3(?=\s)")


def command_is_safe(command: str) -> bool:
    # Preserve the fact that this is a scoped workspace path.  Removing the
    # prefix turned ``ls /workspace/`` into ``ls /`` and falsely triggered the
    # host-root scan guard.
    visible = re.sub(r"/workspace(?=/|\s|$|['\"])", "workspace", command)
    return not any(
        pattern.search(visible)
        for pattern in (
            NETWORK_PATTERN,
            DESTRUCTIVE_PATTERN,
            ESCAPE_PATTERN,
            PYTHON_NETWORK_PATTERN,
            ROOT_SCAN_PATTERN,
        )
    )


def extract_table_names(command: str) -> list[str]:
    return sorted({match.group(1).lower() for match in TABLE_PATTERN.finditer(command)})


def route_sqlite_cli(command: str) -> str:
    """Route the missing image executable to the project read-only module."""
    return SQLITE_COMMAND_PATTERN.sub("python3 -m llin_verl.pi_sqlite_cli", command)
