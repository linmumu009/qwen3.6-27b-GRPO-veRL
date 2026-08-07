from dataclasses import dataclass
from types import SimpleNamespace

from scripts.configure_live_optimizer_checkpoint import (
    TARGET_CONTENTS,
    configure_worker_checkpoint_contents,
)


@dataclass(frozen=True)
class FrozenCheckpointConfig:
    save_contents: list[str]


def test_live_checkpoint_configuration_adds_optimizer(monkeypatch):
    manager = SimpleNamespace(
        checkpoint_save_contents=["model", "extra"],
        checkpoint_config={"save_contents": ["model", "extra"]},
    )
    worker = SimpleNamespace(
        worker_dict={"Actor": SimpleNamespace(engine=SimpleNamespace(checkpoint_mananager=manager))}
    )
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WG_PREFIX", "abcdef")

    inspected = configure_worker_checkpoint_contents(worker, apply=False)
    assert inspected["engines"][0]["after"] == ["model", "extra"]
    assert manager.checkpoint_config["save_contents"] == ["model", "extra"]

    updated = configure_worker_checkpoint_contents(worker, apply=True)
    assert updated["rank"] == 7
    assert updated["engines"][0]["after"] == TARGET_CONTENTS
    assert manager.checkpoint_config["save_contents"] == ["model", "extra"]


def test_live_checkpoint_configuration_leaves_frozen_source_config_untouched(monkeypatch):
    checkpoint_config = FrozenCheckpointConfig(save_contents=["model", "extra"])
    manager = SimpleNamespace(
        checkpoint_save_contents=["model", "extra"],
        checkpoint_config=checkpoint_config,
    )
    worker = SimpleNamespace(
        worker_dict={"Actor": SimpleNamespace(engine=SimpleNamespace(checkpoint_mananager=manager))}
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WG_PREFIX", "abcdef")

    updated = configure_worker_checkpoint_contents(worker, apply=True)

    assert updated["engines"][0]["after"] == TARGET_CONTENTS
    assert checkpoint_config.save_contents == ["model", "extra"]
