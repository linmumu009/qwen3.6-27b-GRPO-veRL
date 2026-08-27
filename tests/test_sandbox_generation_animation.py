from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_sandbox_generation_animation import (
    CONNECTIONS,
    STAGES,
    main,
    render_html,
)


def test_animation_uses_latest_branched_stage_graph_and_boundary() -> None:
    assert [stage.key for stage in STAGES] == [
        "brief",
        "scene",
        "prd",
        "factor",
        "taxonomy",
        "schema",
        "data",
        "tasks",
        "knowledge_schema",
        "documents",
        "knowledge_tasks",
        "hybrid",
        "freeze",
        "rollout",
    ]
    edge_set = {(edge.from_key, edge.to_key, edge.route) for edge in CONNECTIONS}
    assert ("taxonomy", "schema", "auto") in edge_set
    assert ("taxonomy", "knowledge_schema", "right-rail") in edge_set
    assert ("tasks", "hybrid", "merge-left") in edge_set
    assert ("knowledge_tasks", "hybrid", "merge-left") in edge_set
    assert STAGES[-2].title == "三层防泄露冻结"
    assert STAGES[-1].boundary == "external"
    assert "只消费 runner 层" in STAGES[-1].subtitle


def test_animation_is_self_contained_and_exposes_real_artifacts() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert '<script src="http' not in html
    assert '<link href="http' not in html
    assert "requestAnimationFrame" in html
    assert 'aria-live="polite"' in html
    assert 'type="range"' in html
    assert "sandbox_registry.jsonl" in html
    assert "database/*.sqlite" in html
    assert "knowledge_tasks/tasks.jsonl" in html
    assert "hybrid_tasks/tasks.jsonl" in html
    assert "*_tasks.jsonl hidden" in html
    assert "EvidencePlan 同源生成" in html
    assert "GitHub main 7589170" in html
    assert "raw、runner、registry 三层冻结" in html
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


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "sandbox-animation.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
