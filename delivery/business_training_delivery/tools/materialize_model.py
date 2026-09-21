"""Replace the delivery-only model symlink with a verified standalone copy."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "model" / "qwen3.6-27b"
    if not target.is_symlink():
        raise SystemExit("Model is already materialized, or the expected link is missing")
    source = target.resolve(strict=True)
    staging = target.with_name(".qwen3.6-27b-copy")
    if staging.exists():
        raise FileExistsError(staging)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = set(index["weight_map"].values())
    for name in weights:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Missing/unsafe model shard: {name}")
    files = [path for path in source.iterdir() if path.is_file() and
             (path.name in weights or path.suffix in {".json", ".jinja", ".model", ".txt", ".md", ".py"}
              or path.name.startswith("LICENSE"))]
    size = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(target.parent).free < size + 90_000_000_000:
        raise RuntimeError("Insufficient disk space for model copy and training checkpoint")
    staging.mkdir()
    manifest = {}
    for path in sorted(files):
        destination = staging / path.name
        digest = hashlib.sha256()
        with path.open("rb") as src, destination.open("xb") as dst:
            while chunk := src.read(16 * 1024 * 1024):
                dst.write(chunk)
                digest.update(chunk)
        if destination.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"Size mismatch: {path.name}")
        verify = hashlib.sha256()
        with destination.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                verify.update(chunk)
        if verify.hexdigest() != digest.hexdigest():
            raise RuntimeError(f"Digest mismatch: {path.name}")
        manifest[path.name] = {"bytes": destination.stat().st_size, "sha256": digest.hexdigest()}
        print(path.name, "verified", flush=True)
    (staging / "delivery_checksums.json").write_text(json.dumps(manifest, indent=2))
    # Unlink only the delivery symlink. Never alter the source model directory.
    assert target.is_symlink() and target.resolve() == source
    target.unlink()
    staging.rename(target)
    print(f"Standalone model ready: {target}")


if __name__ == "__main__":
    main()
