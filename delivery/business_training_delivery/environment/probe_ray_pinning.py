"""Small diagnostic for Ray/Ascend visible-device host-pinning behavior."""
import json
import os
from pathlib import Path

import ray


def main():
    records = []
    for disable_remap in (False, True):
        if disable_remap:
            os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
        else:
            os.environ.pop("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", None)
        ray.init(address="local", num_cpus=32, include_dashboard=False)

        @ray.remote(num_cpus=1, resources={"NPU": 1})
        class Probe:
            def run(self):
                import torch
                import torch_npu
                from torch_npu.contrib import transfer_to_npu  # noqa: F401
                ids = ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
                record = {"no_remap": disable_remap, "assigned": ids,
                          "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                          "cap_eff": next(x for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("CapEff:"))}
                try:
                    torch_npu.npu.set_device(int(ids[0]) if disable_remap else 0)
                    value = torch.empty(1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=True)
                    record["success"] = True
                    del value
                except Exception as exc:
                    record.update(success=False, error=str(exc)[:500])
                return record

        actors = [Probe.remote() for _ in range(4)]
        result = ray.get([actor.run.remote() for actor in actors], timeout=180)
        records.extend(result)
        print(json.dumps(result), flush=True)
        ray.shutdown()
    Path("/delivery/environment/ray_pinning_probe.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
