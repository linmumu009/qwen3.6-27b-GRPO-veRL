"""Render private knowledge documents and deduplicated research candidates.

Never exports a train.parquet or approves claims merely because retrieval matched.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dump(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    out = args.directory
    topics = json.loads((out / "topic_evidence.private.json").read_text(encoding="utf-8"))
    notes = json.loads((out / "concept_notes.private.json").read_text(encoding="utf-8"))
    pages = {r["id"]: r for r in read(out / "book_pages.private.jsonl")}
    wiki = {r["requested_title"]: r for r in read(out / "wikipedia.private.jsonl")}
    authorities = json.loads((out / "authority_sources.safe.json").read_text(encoding="utf-8"))
    issues = {r["case_index"]: r for r in read(out / "issues.private.jsonl")}
    cases = read(args.cases)
    relations = read(out / "relations.private.jsonl")
    assert set(notes) == {t["id"] for t in topics}
    assert {i for t in topics for i in t["case_indices"]} == set(range(300))
    assert all(title in wiki for t in topics for title in t["wikipedia_titles"])
    links = []
    for i, case in enumerate(cases):
        related = [t for t in topics if i in t["case_indices"]]
        links.append({"case_index": i, "item_hash": case["item_hash"], "dataset": case["dataset"],
                      "source_id": case["source_id"], "topics": [t["id"] for t in related],
                      "book_candidate_ids": sorted({b["page_id"] for t in related for b in t["book_candidates"]}),
                      "wikipedia_revision_ids": sorted({wiki[w]["revision_id"] for t in related for w in t["wikipedia_titles"] if wiki[w].get("revision_id")}),
                      "issues": issues.get(i, {}).get("flags", []),
                      "coverage_verdict": "topic_mapped_not_answer_or_exhaustive_knowledge_coverage_verified"})
    dump(out / "case_knowledge_map.private.jsonl", links)
    intro = ("# 300道错题相关知识：十三份材料整理\n\n"
             "本地研究文档，2026-09-11。所有300题均已登记到62个主题。这里区分整理者的概念关系导读、原书证据与尚待核对的候选。"
             "检索覆盖13份材料；未宣称每道题所需知识均已在教材中找到，也未宣称候选页的全部内容都相关。"
             "原书短标题、定义、条件和公式文本不因长度而删除。解析公式和图形仍需对照原页。\n\n"
             "概念导读是跨来源的编辑整理，不能当作13份教材的逐字引文；补充来源在第二份文档。"
             "本次产物是知识材料与审核清单，不是已获准自动加载的训练包。\n\n")
    book = [intro, "## 主题目录\n\n"]
    for topic in topics:
        book.append(f"- [{topic['title']}](#topic-{topic['id']})\n")
    for topic in topics:
        book.append(f"\n<a id=\"topic-{topic['id']}\"></a>\n\n## {topic['title']}\n\n")
        book.append(notes[topic["id"]]["text"] + "\n\n")
        topic_relations = [r for r in relations if r["topic"] == topic["id"]]
        if topic_relations:
            book.append("| 概念 | 关系 | 关联概念 | 条件或语境 |\n|---|---|---|---|\n")
            for relation in topic_relations:
                book.append("| " + " | ".join(relation[k] for k in ("subject", "relation", "object", "scope")) + " |\n")
            book.append("\n")
        book.append("关联题数：" + str(len(topic["case_indices"])) + "。概念关系与边界见上文；以下为来源索引。\n\n")
        selected = [r for r in topic["book_candidates"] if "editor_selected_source_context" in r["via"]]
        if selected:
            book.append("**优先核对的原书页**（人工指定上下文；仍不表示该页所有段落均已通过事实审查）：\n\n")
            for ref in selected:
                page = pages[ref["page_id"]]
                book.append(f"- [{page['source_file']} · {page['locator']}](#book-{page['id']})\n")
        else:
            book.append("**尚无人工指定的直接教材页**；以下是搜索候选，不能用候选数量表示已覆盖。\n\n")
        book.append("\n**其他相关候选页：**\n\n")
        for ref in topic["book_candidates"]:
            if ref in selected:
                continue
            page = pages[ref["page_id"]]
            book.append(f"- [{page['source_file']} · {page['locator']}](#book-{page['id']})（词汇相关度 {ref['score']:.3f}）\n")
        linked_issues = [issues[i] for i in topic["case_indices"] if i in issues]
        if linked_issues:
            book.append("\n需要核对：" + "；".join(sorted({f for r in linked_issues for f in r["flags"]})) + "。\n")
    book.append("\n## 原书证据正文\n\n以下按原书物理页或ZIP成员定位，保留原英文材料；不是题目与答案拼接。页面可能包含背景段落。\n")
    for page in sorted(pages.values(), key=lambda p: (p["source_file"], str(p["locator"]).zfill(8))):
        book.append(f"\n<a id=\"book-{page['id']}\"></a>\n\n### {page['source_file']} · {page['locator']}\n\n")
        book.append("来源SHA256：" + page["source_id"] + "。\n\n")
        if page.get("additional_pages"):
            book.append("解析器合并的续页：" + str(page["additional_pages"]) + "。\n\n")
        book.append("原处理标记（仅提示，不据此丢掉定义）：" + ", ".join(page["original_flags"]) + "。\n\n")
        book.append(page["text"] + "\n")
    (out / "01_textbook_knowledge.md").write_text("".join(book), encoding="utf-8")
    external = ["# 维基百科与权威来源补充\n\n2026-09-11。百科条目按主题检索，保存版本ID与来源；标题跳转到同一文章时只保留一个正文。"
                "百科文本为署名节选/编排，Wikipedia contributors，依CC BY-SA条款保留来源和版本；不为原13份材料更改许可。"
                "百科与官方资料的描述可能有版本或用途差异，不能只因来自百科就视为无冲突。\n\n",
                "## 其他路径：官方标准、监管定义、作者教材与产品文档\n\n"]
    for source in authorities:
        external.append(f"### {source['title']}\n\n{source['summary']}\n\n[原始来源]({source['url']})。适用主题：{', '.join(source['topics'])}。\n\n")
    external.append("## 维基百科定义资料\n\n许可说明：[Wikipedia Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights)。以下仅使用已取得的正文，部分为导言、部分含扩展段落；不声称下载了每篇全文。\n\n")
    seen = set()
    valid_wiki = []
    for row in wiki.values():
        if row["missing"] or row["disambiguation"] or not row["text"] or row["revision_id"] in seen:
            continue
        seen.add(row["revision_id"])
        valid_wiki.append(row)
        external.append(f"### {row['title']}\n\n[页面]({row['url']}) · [版本 {row['revision_id']}]({row['permalink']})\n\n")
        external.append("Wikipedia contributors；保留原英文，版面已转为纯文本。公式显示如有碎片须核对原网页，未自动作为训练公式。\n\n" + row["text"] + "\n\n")
    (out / "02_external_knowledge.md").write_text("".join(external), encoding="utf-8")
    issue_doc = ["# 冲突、缺图与适用条件核对表\n\n本表保留私有题目供审查，不进入训练正文。列出的事项包含确认的定义/公式冲突、疑似标注问题、语境要求和缺图；不能把它们全部称为错标。未被标记的题也不等于已证明知识完整或与所有来源一致。\n\n"]
    for index, row in sorted(issues.items()):
        case = cases[index]
        issue_doc.append(f"## {case['dataset']} · {case['source_id']}\n\n类型：{', '.join(row['flags'])}\n\n")
        issue_doc.append(case["question"] + "\n\n原标注选项：" + " | ".join(str(case["options"][i]) for i in case["expected"]) + "\n\n")
        issue_doc.extend("- " + note + "\n" for note in row["notes"])
        for aid in row["authority_refs"]:
            source = next(a for a in authorities if a["id"] == aid)
            issue_doc.append(f"- 核对来源：[{source['title']}]({source['url']})\n")
        issue_doc.append("\n处理：保留相关原始材料，不依据未解决的答案标签制造训练事实。\n\n")
    (out / "03_conflicts_and_gaps.md").write_text("".join(issue_doc), encoding="utf-8")
    candidates, seen_text = [], {}
    page_topics = defaultdict(list)
    for topic in topics:
        for ref in topic["book_candidates"]:
            page_topics[ref["page_id"]].append(topic["id"])
    for page in pages.values():
        # Entire pages preserve the context needed to repair short definitions.
        # Training-length packing is intentionally not implied at research stage.
        text = page["text"]
        key = digest(" ".join(text.split()))
        if key in seen_text:
            previous = seen_text[key]
            previous.setdefault("additional_source_occurrences", []).append({
                "source_file": page["source_file"], "source_id": page["source_id"],
                "locator": page["locator"], "block_ids": page["block_ids"]})
            previous["topics"] = sorted(set(previous["topics"]) | set(page_topics[page["id"]]))
            continue
        candidates.append({"id": "llin-book-" + page["id"], "text": text, "text_sha256": digest(text),
                           "kind": "user_supplied_book_context", "topics": sorted(set(page_topics[page["id"]])),
                           "source_file": page["source_file"], "source_id": page["source_id"], "locator": page["locator"],
                           "block_ids": page["block_ids"], "review_flags": page["original_flags"],
                           "training_ready": False})
        seen_text[key] = candidates[-1]
    for row in valid_wiki:
        text = row["text"]
        key = digest(" ".join(text.split()))
        if key in seen_text:
            seen_text[key].setdefault("additional_source_occurrences", []).append({
                "source_url": row["url"], "permalink": row["permalink"],
                "license": row["license"], "attribution": row["attribution"]})
            continue
        candidates.append({"id": "llin-wiki-" + str(row["revision_id"]), "text": text,
                           "text_sha256": digest(text), "kind": "wikipedia_definition_material",
                           "topics": sorted({t["id"] for t in topics for title in t["wikipedia_titles"] if wiki[title].get("revision_id") == row["revision_id"]}),
                           "source_url": row["url"], "permalink": row["permalink"],
                           "license": row["license"], "license_url": row["license_url"],
                           "attribution": row["attribution"], "training_ready": False})
        seen_text[key] = candidates[-1]
    dump(out / "candidate_knowledge.private.jsonl", candidates)
    dump(out / "concept_outlines.private.jsonl", [{"id": t["id"], "title": t["title"], **notes[t["id"]]} for t in topics])
    summary = json.loads((out / "collection.safe.json").read_text(encoding="utf-8"))
    summary.update({"wikipedia_requested_titles": len(wiki), "wikipedia_unique_valid_revisions": len(valid_wiki),
                    "authority_source_records": len(authorities), "cases_with_review_issues": len(issues),
                    "editorial_relation_records": len(relations),
                    "review_issue_counts": dict(Counter(f for r in issues.values() for f in r["flags"])),
                    "deduplicated_research_candidates": len(candidates),
                    "candidate_characters": sum(len(r["text"]) for r in candidates),
                    "training_ready": False, "training_launched": False,
                    "source_family_exclusions": False, "benchmark_text_used_as_training_examples": False,
                    "all_300_cases_have_traceability": len(links) == 300,
                    "methodology": "Known-benchmark-targeted knowledge collection; these benchmarks cannot independently establish generalization after this targeting."})
    files = ["01_textbook_knowledge.md", "02_external_knowledge.md", "03_conflicts_and_gaps.md", "candidate_knowledge.private.jsonl", "case_knowledge_map.private.jsonl"]
    summary["artifacts"] = {name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(), "bytes": (out / name).stat().st_size} for name in files}
    (out / "delivery.safe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
