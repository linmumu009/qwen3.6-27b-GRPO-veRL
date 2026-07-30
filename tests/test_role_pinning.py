import importlib.util
from pathlib import Path


def load_sitecustomize_without_pythonpath():
    path = Path(__file__).parents[1] / "runtime" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("llin_role_sitecustomize", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_accelerator_mapping(monkeypatch):
    module = load_sitecustomize_without_pythonpath()
    monkeypatch.setenv("LLIN_TRAINER_RESOURCE", "trainer-tag")
    monkeypatch.setenv("LLIN_ROLLOUT_RESOURCE", "rollout-tag")
    assert module.accelerator_for_pool("trainer_pool") == "trainer-tag"
    assert module.accelerator_for_pool("rollout_pool_0") == "rollout-tag"
    assert module.accelerator_for_pool("rollout_pool_reward_0") is None
