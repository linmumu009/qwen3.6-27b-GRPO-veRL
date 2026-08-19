import importlib.util
from pathlib import Path
import sys
import types


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


def test_role_pinning_checks_capacity_before_eager_pool_reservation(monkeypatch):
    module = load_sitecustomize_without_pythonpath()
    events = []

    class DummyRayResourcePool:
        def __init__(self, **kwargs):
            events.append(("reserve", kwargs["name_prefix"]))

    class DummyResourcePoolManager:
        _llin_role_pinning = False

        def __init__(self):
            self.resource_pool_spec = {"rollout_pool_0": [16]}
            self.max_colocate_count = 2
            self.resource_pool_dict = {}

        def _check_resource_available(self):
            events.append(("check", None))

    base = types.ModuleType("verl.single_controller.ray.base")
    base.RayResourcePool = DummyRayResourcePool
    base.ResourcePoolManager = DummyResourcePoolManager
    for name in ("verl", "verl.single_controller", "verl.single_controller.ray"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", base)

    module.apply_role_pinning()
    manager = DummyResourcePoolManager()
    manager.create_resource_pool()

    assert events == [("check", None), ("reserve", "rollout_pool_0")]
