from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_logistics_sandbox_example_animation import (
    CONNECTIONS,
    STAGES,
    main,
    render_html,
)


def test_replay_preserves_real_v20_stage_graph_and_timing() -> None:
    assert [stage.key for stage in STAGES] == [
        "input",
        "prd",
        "factor",
        "taxonomy",
        "schema",
        "data",
        "dwh_tasks",
        "catalog",
        "documents",
        "kb_tasks",
        "hybrid",
        "freeze",
        "rollout",
    ]
    assert max(STAGES, key=lambda stage: stage.duration_seconds).key == "factor"
    assert next(stage for stage in STAGES if stage.key == "factor").duration_seconds == 3074
    assert "80:31" in next(stage for stage in STAGES if stage.key == "freeze").step
    edges = {(edge.from_key, edge.to_key, edge.route) for edge in CONNECTIONS}
    assert ("taxonomy", "schema", "auto") in edges
    assert ("taxonomy", "catalog", "right-rail") in edges
    assert ("dwh_tasks", "hybrid", "merge-left") in edges
    assert ("kb_tasks", "hybrid", "merge-left") in edges
    assert STAGES[-1].boundary == "external"


def test_replay_embeds_only_safe_aggregate_evidence() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert '<script src="http' not in html
    assert '<link href="http' not in html
    assert "requestAnimationFrame" in html
    assert "物流运营沙箱 · 真实生成重放" in html
    assert "运营分析-8767b626" in html
    assert "36,101 条记录" in html
    assert "65 个 Markdown" in html
    assert "555 条最终任务" in html
    assert "465 条任务" in html
    assert "data→policy 171" in html
    assert "policy→data 156" in html
    assert "compliance 173" in html
    assert "hidden gold 不展示" in html
    assert "gold_answer" not in html
    assert "SELECT " not in html
    assert "__STAGE_DATA__" not in html
    assert "__CONNECTION_DATA__" not in html
    assert "__SOURCE_DATA__" not in html

    stage_match = re.search(
        r'<script id="stageData" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert stage_match is not None
    embedded = json.loads(stage_match.group(1))
    assert [stage["key"] for stage in embedded] == [stage.key for stage in STAGES]
    assert all(stage["replay_ms"] > 0 for stage in embedded)


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "logistics-sandbox-replay.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
