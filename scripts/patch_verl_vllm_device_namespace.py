"""Map Ascend physical devices into the rollout parent's local IPC namespace."""
from pathlib import Path


def patch(root: Path) -> None:
    server = root / 'workers/rollout/vllm_rollout/vllm_async_server.py'
    utils = root / 'workers/rollout/vllm_rollout/utils.py'
    old_server = '        os.environ[get_visible_devices_keyword()] = cuda_visible_devices\n'
    new_server = old_server + '        os.environ["VERL_ROLLOUT_LOCAL_DEVICES"] = cuda_visible_devices\n'
    old_rank = '            return int(visible_ranks[worker_local_rank % len(visible_ranks)])'
    new_rank = '''            physical = visible_ranks[worker_local_rank % len(visible_ranks)]
            parent = os.environ.get("VERL_ROLLOUT_LOCAL_DEVICES")
            if parent is not None:
                devices = [item.strip() for item in parent.split(",") if item.strip()]
                if len(set(devices)) != len(devices) or physical not in devices:
                    raise ValueError("Invalid rollout parent device namespace")
                return devices.index(physical)
            return int(physical)'''
    changes = []
    for path, old, new in [(server, old_server, new_server), (utils, old_rank, new_rank)]:
        text = path.read_text(encoding='utf-8')
        if new in text:
            continue
        if text.count(old) != 1:
            raise ValueError(f'Expected one patch anchor in {path}')
        changes.append((path, text.replace(old, new, 1)))
    for path, text in changes:
        path.write_text(text, encoding='utf-8')
