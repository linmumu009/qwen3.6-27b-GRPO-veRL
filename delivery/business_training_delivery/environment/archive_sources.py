"""Run on machine05 host to snapshot tracked source files from the known container.

Copies working-tree content (including tracked patches), never container homes,
environment variables, credentials, run directories, or the whole writable layer.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "llin-verl-trainer-m05-20260730"
SOURCES = {
    "verl": "/verl",
    "vllm": "/vllm",
    "vllm-ascend": "/vllm-ascend",
    "Megatron-Bridge": "/workspace/llin-verl-grpo/reference/Megatron-Bridge-de93536e",
}


def inside(*args):
    return subprocess.check_output(["docker", "exec", CONTAINER, *args])


def main():
    records = {}
    for name, source in SOURCES.items():
        destination = ROOT / "frameworks" / name
        if destination.exists():
            raise FileExistsError(destination)
        names = inside("git", "-C", source, "ls-files", "-z")
        # Archive only tracked files that still exist in the working tree.
        command = ["docker", "exec", "-i", CONTAINER, "tar", "-C", source,
                   "--null", "-T", "-", "-czf", "-"]
        archive = ROOT / "frameworks" / (name + ".tar.gz")
        with archive.open("wb") as stream:
            subprocess.run(command, input=names, stdout=stream, check=True)
        destination.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                resolved = (destination / member.name).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError(member.name)
                if member.issym() or member.islnk():
                    target = (resolved.parent / member.linkname).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError(f"External source link: {member.name}")
            tar.extractall(destination)
        patch = inside("git", "-C", source, "diff", "HEAD", "--")
        (ROOT / "frameworks" / (name + ".patch")).write_bytes(patch)
        records[name] = {
            "commit": inside("git", "-C", source, "rev-parse", "HEAD").decode().strip(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "tracked_files": len(names.rstrip(b"\0").split(b"\0")),
        }
        print(name, records[name], flush=True)
    (ROOT / "environment" / "source_manifest.json").write_text(json.dumps(records, indent=2))
    packages = inside("python3", "-m", "pip", "list", "--format=json").decode()
    (ROOT / "environment" / "packages.json").write_text(json.dumps(json.loads(packages), indent=2))


if __name__ == "__main__":
    main()
