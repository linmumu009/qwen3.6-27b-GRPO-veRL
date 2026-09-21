"""Convert the shipped, inspectable JSONL datasets into veRL Parquet files."""
import json
from pathlib import Path

TYPES = ["SFT", "trajory-SFT", "GRPO", "agentic-GRPO", "CPT"]


def prepare(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    counts = {}
    for kind in TYPES:
        counts[kind] = {}
        for split in ("train", "val"):
            source = root / "datasets" / kind / (split + ".jsonl")
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            if not rows:
                raise ValueError(f"Empty dataset: {source}")
            pq.write_table(pa.Table.from_pylist(rows), source.with_suffix(".parquet"))
            assert pq.read_table(source.with_suffix(".parquet")).num_rows == len(rows)
            counts[kind][split] = len(rows)
    (root / "datasets" / "manifest.json").write_text(json.dumps(counts, indent=2))
    return counts


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[1]))
