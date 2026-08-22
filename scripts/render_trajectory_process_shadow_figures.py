#!/usr/bin/env python3
"""Render dependency-free SVG figures for the safe shadow report."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#1F2937"
GRID = "#D1D5DB"
BACKGROUND = "#FFFFFF"


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        '<style>text{font-family:Arial,"Noto Sans",sans-serif;fill:#1F2937}.title{font-size:18px;font-weight:700}.label{font-size:13px}.small{font-size:12px;fill:#6B7280}.value{font-size:12px;font-weight:700}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8", newline="\n")


def reward_figure(summary: dict, output: Path) -> None:
    width, height = 760, 420
    left, right, top, bottom = 86, 28, 64, 76
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = 1.35
    rewards = summary["reward_distributions"]
    categories = [
        ("Correct", rewards["old_boss_shadow"]["correct"], rewards["new"]["correct"]),
        ("Incorrect", rewards["old_boss_shadow"]["incorrect"], rewards["new"]["incorrect"]),
    ]
    body = [
        '<text class="title" x="24" y="32">New reward restores outcome separation</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="{INK}"/>',
    ]
    for tick in (0.0, 0.3, 0.6, 0.9, 1.2):
        y = top + plot_height - tick / maximum * plot_height
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
                f'<text class="small" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>',
            ]
        )
    group_centers = [left + plot_width * 0.28, left + plot_width * 0.72]
    bar_width = 82
    for center, (label, old, new) in zip(group_centers, categories, strict=True):
        for x, metric, color in (
            (center - bar_width - 5, old, ORANGE),
            (center + 5, new, BLUE),
        ):
            bar_height = metric["mean"] / maximum * plot_height
            y = top + plot_height - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="2" fill="{color}"/>'
            )
            body.append(
                f'<text class="value" x="{x + bar_width / 2:.1f}" y="{max(top + 12, y - 8):.1f}" text-anchor="middle">{metric["mean"]:.3f}</text>'
            )
        body.append(
            f'<text class="label" x="{center:.1f}" y="{top + plot_height + 28}" text-anchor="middle">{escape(label)}</text>'
        )
    body.extend(
        [
            f'<rect x="{left + 16}" y="{height - 28}" width="13" height="13" fill="{ORANGE}"/><text class="small" x="{left + 35}" y="{height - 17}">Boss reward_total shadow</text>',
            f'<rect x="{left + 240}" y="{height - 28}" width="13" height="13" fill="{BLUE}"/><text class="small" x="{left + 259}" y="{height - 17}">New gated reward</text>',
        ]
    )
    save_svg(output, width, height, body)


def component_figure(summary: dict, output: Path) -> None:
    width, height = 820, 490
    left, right, top = 166, 40, 68
    plot_width = width - left - right
    distributions = summary["process_component_distributions"]
    components = [
        ("SQL result", "process_sql"),
        ("Required tables", "process_table"),
        ("Required fields", "process_field"),
        ("DWH execution", "process_fit"),
        ("Efficiency", "process_efficiency"),
        ("Weighted P", "process_score"),
    ]
    row_height, bar_height = 58, 18
    body = [
        '<text class="title" x="24" y="32">Process evidence is informative but not outcome-defining</text>'
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_width
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + row_height * len(components)}" stroke="{GRID}"/>',
                f'<text class="small" x="{x:.1f}" y="{top + row_height * len(components) + 22}" text-anchor="middle">{tick:.2f}</text>',
            ]
        )
    for index, (label, key) in enumerate(components):
        center_y = top + index * row_height + 24
        correct = distributions[key]["correct"]["mean"]
        incorrect = distributions[key]["incorrect"]["mean"]
        body.append(
            f'<text class="label" x="{left - 14}" y="{center_y + 5}" text-anchor="end">{escape(label)}</text>'
        )
        for y, value, color in (
            (center_y - bar_height - 2, correct, BLUE),
            (center_y + 2, incorrect, ORANGE),
        ):
            body.append(
                f'<rect x="{left}" y="{y}" width="{value * plot_width:.1f}" height="{bar_height}" rx="2" fill="{color}"/>'
            )
    body.extend(
        [
            f'<rect x="{left}" y="{height - 26}" width="13" height="13" fill="{BLUE}"/><text class="small" x="{left + 19}" y="{height - 15}">Correct</text>',
            f'<rect x="{left + 100}" y="{height - 26}" width="13" height="13" fill="{ORANGE}"/><text class="small" x="{left + 119}" y="{height - 15}">Incorrect</text>',
        ]
    )
    save_svg(output, width, height, body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    reward_figure(summary, args.output_dir / "qwen38_shadow_reward_separation.svg")
    component_figure(summary, args.output_dir / "qwen38_shadow_process_components.svg")


if __name__ == "__main__":
    main()
