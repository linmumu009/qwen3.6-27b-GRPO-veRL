"""Opt-in Ray resource-pool pinning for separated veRL roles.

Python imports this module automatically when ``runtime`` is on ``PYTHONPATH``.
The patch is inert unless ``LLIN_PIN_RAY_ROLES=1`` is set.
"""

from __future__ import annotations

import os


def accelerator_for_pool(pool_name: str) -> str | None:
    if pool_name == "trainer_pool":
        return os.getenv("LLIN_TRAINER_RESOURCE", "llin_trainer")
    if pool_name.startswith("rollout_pool_") and "reward" not in pool_name and "teacher" not in pool_name:
        return os.getenv("LLIN_ROLLOUT_RESOURCE", "llin_rollout")
    return None


def apply_role_pinning() -> None:
    from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager

    if getattr(ResourcePoolManager, "_llin_role_pinning", False):
        return

    def create_resource_pool(self: ResourcePoolManager) -> None:
        # RayResourcePool creates placement groups eagerly. Checking available
        # accelerators afterwards races with those reservations and can report
        # zero free NPUs even though this call reserved them itself. Validate
        # capacity first; Ray scheduling remains the final allocation gate.
        self._check_resource_available()
        for pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=self.max_colocate_count,
                name_prefix=pool_name,
                accelerator_type=accelerator_for_pool(pool_name),
            )
            self.resource_pool_dict[pool_name] = resource_pool

    ResourcePoolManager.create_resource_pool = create_resource_pool
    ResourcePoolManager._llin_role_pinning = True


if os.getenv("LLIN_PIN_RAY_ROLES") == "1":
    apply_role_pinning()
