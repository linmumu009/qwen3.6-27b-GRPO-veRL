"""Exercise live HCCL training and two evaluation groups on the same 16 NPUs."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.train_logistics_wrong_text_sft import evaluation_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', timeout=timedelta(seconds=600))
    value = torch.ones(1, device=f'npu:{rank}')
    dist.all_reduce(value)
    assert value.item() == dist.get_world_size()
    result = [None]
    if not args.child and dist.get_rank() == 0:
        children = []
        try:
            for index, ports in enumerate([('43000-43199', '44000-44199'), ('45000-45199', '46000-46199')]):
                env = evaluation_environment(os.environ)
                env.update(ASCEND_RT_VISIBLE_DEVICES=','.join(str(i) for i in range(index * 8, index * 8 + 8)),
                    HCCL_HOST_SOCKET_PORT_RANGE=ports[0], HCCL_NPU_SOCKET_PORT_RANGE=ports[1])
                children.append(subprocess.Popen([sys.executable, '-m', 'torch.distributed.run',
                    '--standalone', '--nnodes=1', '--nproc_per_node=8', __file__, '--child'], env=env))
            codes = [process.wait(timeout=450) for process in children]
            if codes != [0, 0]:
                raise RuntimeError(f'concurrent HCCL test failed: {codes}')
            run = Path(os.environ['WRONG_TEXT_RUN'])
            (run / 'ports.safe.json').write_text(json.dumps(dict(passed=True, training_world_size=16,
                concurrent_evaluation_world_sizes=[8, 8], physical_npus=16), indent=2) + '\n')
        except Exception as exc:
            result[0] = str(exc)
        finally:
            for process in children:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
    if not args.child:
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
