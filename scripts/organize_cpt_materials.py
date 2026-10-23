"""Copy historical CPT sources and processed data into a verified local archive.

The archive contains private text and stays outside Git. The remote download
directory must already contain the files listed in its remote_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


GROUPS = (
    "01_最初物流教材",
    "02_Logistika相关13份教材",
    "03_术语与知识补充",
    "04_其他材料",
)
LAYERS = ("01_原始内容", "02_处理后训练语料")
BOOK_NAMES = (
    ("43_Book_", "01_Ports_and_Waterways.pdf"),
    ("Beyond Lean_", "02_Beyond_Lean_Simulation_in_Practice.pdf"),
    ("challenges-", "03_Challenges_and_Opportunities_in_International_Business.zip"),
    ("Container terminals", "04_Container_Terminals_and_Automated_Transport_Systems.pdf"),
    ("Distribution logistics", "05_Distribution_Logistics.pdf"),
    ("Fundamentals", "06_Fundamentals_of_Global_Strategy.pdf"),
    ("International Trade", "07_International_Trade_Theory_and_Policy.pdf"),
    ("JSI_", "08_JSI_Supply_Chain_Managers_Handbook.pdf"),
    ("Knoop", "09_Traffic_Flow_Theory.pdf"),
    ("KS-GQ-", "10_Glossary_for_Transport_Statistics_2019.pdf"),
    ("Managing supply", "11_Managing_Supply_Chain_Risk.pdf"),
    ("MITx_", "12_MITx_SCM_Key_Concepts.pdf"),
    ("wh-sci", "13_Warehouse_and_Distribution_Science.pdf"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Archive:
    def __init__(self, repo: Path, output: Path, downloads: Path):
        self.repo, self.output, self.downloads = repo, output, downloads
        self.rows: list[dict] = []
        self.remote = {
            row["filename"]: row
            for row in json.loads((downloads / "remote_manifest.json").read_text(encoding="utf-8-sig"))
        }
        self.hashes: dict[Path, str] = {}

    def put(self, source: Path, target: Path, status: str, remote: bool = False) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = self.output / target
        if destination.exists():
            raise FileExistsError(destination)
        digest = self.hashes.setdefault(source, sha256(source))
        origin = str(source.relative_to(self.repo))
        if remote:
            metadata = self.remote[source.name]
            if digest != metadata["sha256"] or source.stat().st_size != metadata["bytes"]:
                raise ValueError(f"Remote download changed: {source.name}")
            origin = metadata["source"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != digest:
            raise ValueError(f"Copy hash mismatch: {target}")
        self.rows.append({
            "path": target.as_posix(), "source": origin, "original_filename": source.name,
            "bytes": source.stat().st_size, "sha256": digest, "status": status,
        })

    def tree(self, source: Path, target: Path, status: str) -> None:
        files = sorted(p for p in source.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix not in {".py", ".pyc"})
        if not files:
            raise ValueError(f"Empty source directory: {source}")
        for path in files:
            self.put(path, target / path.relative_to(source), status)

    def document(self, target: Path, text: str) -> None:
        path = self.output / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        self.rows.append({
            "path": target.as_posix(), "source": "generated:organization_notes",
            "original_filename": "", "bytes": path.stat().st_size,
            "sha256": sha256(path), "status": "整理说明，不是历史语料",
        })

    def build(self) -> None:
        resources = self.repo / "CPT_resources"
        raw = [Path(LAYERS[0]) / group for group in GROUPS]
        processed = [Path(LAYERS[1]) / group for group in GROUPS]
        for folder in raw + processed:
            (self.output / folder).mkdir(parents=True)

        self.put(resources / "The_Handbook_of_Logistics_and_Distribution_Management_8th_Edition.txt",
                 raw[0] / "The_Handbook_of_Logistics_and_Distribution_Management_8th_Edition.txt",
                 "已存教材TXT原文；未找到对应原始PDF")
        for name in ("handbook8e_cpt_4096.jsonl", "handbook8e_cpt_4096.parquet"):
            self.put(self.downloads / name, processed[0] / "01_正式训练语料" / name,
                     "实际用于单书CPT的116条语料", remote=True)
        for name in ("handbook8e_cpt_4096_20260903.safe.json",
                     "qwen36_step120_logistics_single_book_cpt_20260903.md"):
            self.put(self.repo / "docs" / name, processed[0] / "02_来源与训练记录" / name,
                     "历史来源与训练记录")

        originals = list((resources / "CPT-LOGISTS").iterdir())
        if len(originals) != 13:
            raise ValueError("Expected exactly 13 original textbook files")
        for prefix, name in BOOK_NAMES:
            matches = [p for p in originals if p.name.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous book prefix: {prefix}")
            self.put(matches[0], raw[1] / name, "原始教材：12份PDF和1份HTML教材ZIP")
        for source, target, status in (
            ("llin-mineru-results", "01_OCR识别结果_MinerU", "14份历史MinerU识别JSON"),
            ("llin-logists-cache-v1", "02_文本提取中间结果", "历史提取正文，含页码和块来源"),
            ("llin-logists-v5-full-20260911", "03_正式训练包_13份全量", "13份材料正式版；train为4707条，其余为审计或空验证集"),
            ("llin-logists-cpt-v4", "04_历史训练包_9份材料", "历史9份材料版；3016条训练和161条开发验证"),
            ("llin-mineru-input", "05_处理过程资料/OCR拆分输入", "由原始PDF拆出的OCR输入和页面抽查，不是另一套教材原件"),
            ("llin-logists-inspection", "05_处理过程资料/版面抽查", "历史版面质量抽查"),
        ):
            self.tree(resources / source, processed[1] / target, status)
        self.put(resources / "llin-knowledge-complete-20260911/all_source_pages.private.jsonl",
                 processed[1] / "02_文本提取中间结果/all_source_pages.private.jsonl",
                 "13份教材提取后的页级或HTML成员级正文")

        expansion = resources / "llin-knowledge-expansion-20260911"
        complete = resources / "llin-knowledge-complete-20260911"
        for source, target, status in (
            (expansion / "wikipedia.private.jsonl", "01_维基百科采集/wikipedia.private.jsonl", "历史百科版本快照；全文或导言，尚未全部进入训练"),
            (expansion / "authority_sources.safe.json", "02_公开机构来源与片段/authority_sources.safe.json", "23条公开机构来源说明，非23份完整原文件"),
            (complete / "core/sources.private.jsonl", "02_公开机构来源与片段/sources.private.jsonl", "121条来源片段和版本索引，含49条百科记录；不是全部网页或PDF全文"),
            (complete / "iata_codes_2024.ttl", "02_公开机构来源与片段/iata_codes_2024.ttl", "已存IATA代码表"),
            (expansion / "book_pages.private.jsonl", "03_教材候选原文/book_pages.private.jsonl", "402个教材候选页或HTML成员；完整原书在第二类"),
        ):
            self.put(source, raw[2] / target, status)
        for source, target, status in (
            (complete / "core", "01_正式补充知识_205条", "205条正式补充；train为205条，来源文件为追溯信息"),
            (complete, "02_实际整合训练包_13份教材加205条", "实际整合CPT包；release/train为4912条，core及其他文件是补充或追溯信息"),
            (resources / "llin-knowledge-core-v1-20260911", "03_早期补充版本_110条", "早期110条版本，被205条补充版本扩展"),
            (expansion, "04_研究候选与缺口_未全部用于训练", "研究候选和缺口，不可将整个目录当作训练输入"),
        ):
            self.tree(source, processed[2] / target, status)

        erp = resources / "cpt_corpus_v1_0_2_20260910"
        erp_rows = jsonl(erp / "train.jsonl") + jsonl(erp / "validation.jsonl")
        erp_hashes = {row["raw_sha256"] for row in erp_rows}
        raw_paths = {p.stem: p for p in resources.glob("material_survey_20260910*/raw/*")}
        if len(erp_hashes) != 50 or not erp_hashes <= raw_paths.keys():
            raise ValueError("Missing original ERPNext pages")
        for digest in sorted(erp_hashes):
            path = raw_paths[digest]
            self.put(path, raw[3] / "01_ERPNext原始网页" / path.name,
                     "ERPNext正式包对应的50份网页原件；47训练、3开发验证")
        for source in sorted(resources.glob("material_survey_20260910*")):
            if source.name.endswith("_index"):
                continue
            self.tree(source, raw[3] / "02_公开候选资料库" / source.name,
                      "历史公开资料归档；收集不等于全部用于训练")
        for name in ("LogistikaBench.csv", "SC-bench-main.zip"):
            self.put(self.repo / "datasets" / name, raw[3] / "03_题目诊断原始输入" / name,
                     "原始评测数据，仅作题目CPT诊断来源")
        self.put(self.downloads / "frozen_cases_source.jsonl",
                 raw[3] / "03_题目诊断原始输入/frozen_cases_source.jsonl",
                 "诊断实验冻结的1672题输入；经过格式归一化，不冒充原始发布格式", remote=True)
        self.tree(erp, processed[3] / "01_ERPNext正式训练包",
                  "train为47条训练、validation为3条开发验证；其他为来源和候选审计")
        exam = processed[3] / "02_题目CPT诊断_非独立评测语料"
        for folder, names in (
            ("01_原题正式均衡版", ("direct-balanced.items.jsonl", "direct-balanced.cpt.parquet", "direct-balanced.corpus.safe.json")),
            ("02_改写题干正式版", ("rewritten-stems.formal.jsonl", "rewritten-balanced.items.jsonl", "rewritten-balanced.cpt.parquet", "rewritten-balanced.corpus.safe.json")),
            ("03_早期未均衡版本", ("direct.items.jsonl", "direct.cpt.parquet", "direct.corpus.safe.json")),
        ):
            for name in names:
                self.put(self.downloads / name, exam / folder / name,
                         "受控题目污染诊断；正式两臂各1672题、64个训练块；早期版本另存", remote=True)
        self.tree(resources / "material_survey_20260910_index",
                  processed[3] / "03_候选资料提取正文_未全部用于训练",
                  "113份PDF的提取文本及抽查，未整体投入训练")
        for name in ("cpt_material_survey_20260910.md", "cpt_corpus_release_20260910.md",
                     "cpt_erp_one_epoch_20260910.md",
                     "qwen36_step120_logistics_exam_cpt_direct_vs_rewritten_20260904.md"):
            self.put(self.repo / "docs" / name, processed[3] / "04_历史说明" / name,
                     "历史调查、制数与训练说明；其中旧相对链接保留")
        self.put(self.downloads / "remote_manifest.json", processed[3] / "04_历史说明/五号机下载来源与指纹.json",
                 "本次远端文件指纹；无连接凭据")
        self.notes(raw, processed)
        self.write_source_index(resources, raw[3])
        with (self.output / "文件清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        with (self.output / "文件清单.private.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def notes(self, raw: list[Path], processed: list[Path]) -> None:
        raw_notes = (
            "本次找到的是原始TXT文本，未找到对应PDF。保留原文，不将TXT转换成伪原版PDF。",
            "12份PDF加1份HTML教材ZIP，均为原文件的实体副本。为方便阅读缩短了文件名，原名、来源和SHA256见总文件清单。",
            "保存历史百科版本、教材候选页、机构来源片段和索引。机构来源部分未保存完整网页或完整PDF，不将片段和来源说明冒充全文。原书全文见第二类。",
            "包括50份ERPNext原始网页、470个公开候选原文件及其下载清单、原始Benchmark CSV和ZIP、诊断实验冻结输入。ERPNext网页与公开候选库中有50份重复，为保留两种查阅方式而复制，不增加独立来源数量。",
        )
        processed_notes = (
            "01_正式训练语料中是从五号机补回的116条JSONL及Parquet，逐字节核对远端指纹。来源与训练记录另存。",
            "依次存放14份MinerU OCR结果、文本提取中间结果、13份全量正式包、9份材料历史版，以及OCR拆分PDF和版面抽查。正式13份训练文件为03_正式训练包_13份全量/train.parquet，共4707条。staging和quarantine不是正式训练输入。",
            "205条正式补充、13份教材加205条的4912条实际整合包、早期110条补充和未全部使用的候选分别保存。实际整合包的release/train.parquet是4912条；它已包含205条补充，不要再次重复拼接。最初那本手册及ERPNext不在该整合包中。",
            "ERPNext正式包含47条训练和3条开发验证；题目诊断两臂分别保存，原题与改写各1672题打包为64块；未均衡历史版本单列。候选PDF提取文本有113份，没有整体用于训练。题目诊断已使用评测答案，不能当作独立测试结果。",
        )
        for folders, descriptions in ((raw, raw_notes), (processed, processed_notes)):
            for folder, description in zip(folders, descriptions):
                self.document(folder / "说明.md", f"# {folder.name}\n\n{description}\n\n所有历史文件均为复制，原位置不变。总文件清单保留来源、状态、大小和SHA256。")
        self.document(Path("整理说明.md"), """# CPT语料整理（2026-10-23）

两个总文件夹，每个均有相同的四个子文件夹。历史文件均为实体副本，未移动或修改原件。

| 类别 | 原始内容 | 处理后训练语料 |
|---|---|---|
| 最初物流教材 | [TXT原文](01_原始内容/01_最初物流教材/) | [116条正式语料](02_处理后训练语料/01_最初物流教材/) |
| Logistika相关13份教材 | [12份PDF和1份ZIP](01_原始内容/02_Logistika相关13份教材/) | [OCR、文本提取、全量及历史训练包](02_处理后训练语料/02_Logistika相关13份教材/) |
| 术语与知识补充 | [百科版本及来源片段](01_原始内容/03_术语与知识补充/) | [205条补充和4912条整合包](02_处理后训练语料/03_术语与知识补充/) |
| 其他材料 | [ERPNext、公开候选库、诊断原始输入](01_原始内容/04_其他材料/) | [ERPNext、题目诊断和候选提取正文](02_处理后训练语料/04_其他材料/) |

原始TXT不冒充原版PDF；机构来源索引和片段不冒充完整网页或完整文档。已有PDF、HTML、ZIP和采集JSON按原样保存。

处理后目录同时保存识别、提取和清洗中间结果，以及实际训练文件。候选、隔离内容和历史版本均作明确区分，不应把整个目录直接作为训练输入。本次没有重新OCR、重写语料或启动训练，也未发现本批材料中有TTS音频产物。

13份教材加205条知识是一个实际整合包，不包含最初的物流手册或ERPNext。整理目录之间存在为追溯保留的副本，不得把文件数当作独立来源数或累加训练量。

文件清单.csv和文件清单.private.jsonl记录每个文件的原位置、原名、大小、SHA256及用途。完整性检查.json记录校验结果。公开资料来源索引在“原始内容/其他材料”内，保留标题、URL和本次归档位置。

正文、原题和下载内容保存在本地，已由.gitignore排除。只提交整理工具和README版本记录。
""")

    def write_source_index(self, resources: Path, target: Path) -> None:
        rows = []
        for manifest in sorted(resources.glob("material_survey_20260910*/manifest.jsonl")):
            for row in jsonl(manifest):
                local = row.get("local_path", "").replace("\\", "/")
                archived = ""
                if local:
                    archived = (target / "02_公开候选资料库" / manifest.parent.name / "raw" / Path(local).name).as_posix()
                rows.append({"title": row.get("title", ""), "url": row.get("url", ""),
                             "status": row.get("status", ""), "sha256": row.get("sha256", ""),
                             "archived_path": archived})
        relative = target / "公开资料来源索引.csv"
        path = self.output / relative
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.rows.append({"path": relative.as_posix(), "source": "generated:original_download_manifests",
                          "original_filename": "", "bytes": path.stat().st_size,
                          "sha256": sha256(path), "status": "公开资料标题与本次归档位置索引"})


def verify(output: Path) -> dict:
    import pyarrow.parquet as pq

    rows = jsonl(output / "文件清单.private.jsonl")
    with (output / "文件清单.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if csv_rows != [{key: str(value) for key, value in row.items()} for row in rows]:
        raise ValueError("CSV and JSONL file inventories differ")
    for row in rows:
        path = output / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Archive integrity mismatch: {row['path']}")
    for layer in LAYERS:
        actual = sorted(p.name for p in (output / layer).iterdir() if p.is_dir())
        if actual != sorted(GROUPS):
            raise ValueError(f"Unexpected group structure: {layer}")
    processed = output / LAYERS[1]
    book = processed / GROUPS[0] / "01_正式训练语料"
    textbooks = processed / GROUPS[1]
    knowledge = processed / GROUPS[2]
    other = processed / GROUPS[3]
    original_books = output / LAYERS[0] / GROUPS[1]
    original_counts = {
        "pdf": len(list(original_books.glob("*.pdf"))),
        "html_zip": len(list(original_books.glob("*.zip"))),
    }
    if original_counts != {"pdf": 12, "html_zip": 1}:
        raise ValueError("Missing original textbook files")
    source_index = output / LAYERS[0] / GROUPS[3] / "公开资料来源索引.csv"
    with source_index.open(encoding="utf-8-sig", newline="") as stream:
        sources = list(csv.DictReader(stream))
    indexed_paths = set()
    for source in sources:
        if source["archived_path"]:
            path = output / source["archived_path"]
            if sha256(path) != source["sha256"]:
                raise ValueError(f"Public source index mismatch: {source['archived_path']}")
            indexed_paths.add(source["archived_path"])
    if len(indexed_paths) != 470:
        raise ValueError("Expected 470 archived public source files")
    checks = [
        (book / "handbook8e_cpt_4096", 116),
        (textbooks / "03_正式训练包_13份全量/train", 4707),
        (textbooks / "04_历史训练包_9份材料/train", 3016),
        (textbooks / "04_历史训练包_9份材料/validation", 161),
        (knowledge / "01_正式补充知识_205条/train", 205),
        (knowledge / "02_实际整合训练包_13份教材加205条/release/train", 4912),
        (knowledge / "03_早期补充版本_110条/train", 110),
        (other / "01_ERPNext正式训练包/train", 47),
        (other / "01_ERPNext正式训练包/validation", 3),
    ]
    datasets = []
    texts = {}
    for stem, count in checks:
        records = jsonl(Path(str(stem) + ".jsonl"))
        table = pq.read_table(Path(str(stem) + ".parquet"), columns=["text"])
        actual = [record["text"] for record in records]
        if len(records) != count or table.column("text").to_pylist() != actual:
            raise ValueError(f"Training content/record count mismatch: {stem}")
        texts[count] = actual
        datasets.append({"path": str(stem.relative_to(output)), "records": count,
                         "jsonl_parquet_text_match": True})
    if texts[4912] != texts[4707] + texts[205]:
        raise ValueError("Combined training corpus differs from textbooks plus supplement")
    for folder, prefix in (("01_原题正式均衡版", "direct-balanced"),
                           ("02_改写题干正式版", "rewritten-balanced")):
        base = other / "02_题目CPT诊断_非独立评测语料" / folder
        items = jsonl(base / f"{prefix}.items.jsonl")
        blocks = pq.read_metadata(base / f"{prefix}.cpt.parquet").num_rows
        if len(items) != 1672 or blocks != 64:
            raise ValueError(f"Exam diagnostic corpus mismatch: {folder}")
        metadata = json.loads((base / f"{prefix}.corpus.safe.json").read_text(encoding="utf-8"))
        frozen = output / LAYERS[0] / GROUPS[3] / "03_题目诊断原始输入/frozen_cases_source.jsonl"
        if sha256(frozen) != metadata["source_sha256"]:
            raise ValueError("Frozen diagnostic source differs from historical training input")
        if sha256(base / f"{prefix}.cpt.parquet") != metadata["parquet_sha256"]:
            raise ValueError("Diagnostic Parquet differs from historical training input")
        datasets.append({"path": str(base.relative_to(output)), "items": 1672, "blocks": 64})
    expected = {row["path"] for row in rows}
    excluded = {"文件清单.csv", "文件清单.private.jsonl", "完整性检查.json"}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual - excluded != expected:
        raise ValueError("Unlisted or missing archive files")
    summary = {
        "status": "PASS", "date": "2026-10-23", "files_verified": len(rows),
        "bytes_verified": sum(row["bytes"] for row in rows),
        "source_files_copied": sum(not row["source"].startswith("generated:") for row in rows),
        "groups": dict(Counter("/".join(row["path"].split("/")[:2]) for row in rows
                               if row["path"].startswith(LAYERS))),
        "original_textbooks": original_counts, "public_source_files_indexed": len(indexed_paths),
        "ocr_json_files": len(list((textbooks / "01_OCR识别结果_MinerU").glob("*.json"))),
        "datasets": datasets, "combined_equals_textbooks_plus_supplement": True,
        "remote_files_sha256_checked_before_copy": True,
    }
    if summary["ocr_json_files"] != 14:
        raise ValueError("Missing MinerU outputs")
    write_json(output / "完整性检查.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("CPT语料整理_20261023"))
    parser.add_argument("--downloads", type=Path, default=Path(".tmp/cpt-materials-download-20261023"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    downloads = (repo / args.downloads).resolve()
    output.relative_to(repo)
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=False)
        Archive(repo, output, downloads).build()
    summary = verify(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
