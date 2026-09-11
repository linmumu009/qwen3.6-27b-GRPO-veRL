"""Fetch public concept titles only; never send benchmark questions or answers.

Preserves revision IDs, attribution and licence links. Snapshots are research
evidence, not automatically approved training text. Requires requests.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8-sig"))
    titles = sorted({title for topic in topics for title in topic["wikipedia_titles"]})
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "LogisticsKnowledgeAudit/1.0 (educational source collection)"
    output_file = args.output / "wikipedia.private.jsonl"
    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()] if output_file.exists() else []
    rows = [r for r in rows if r["text"] and not r["disambiguation"] and r["requested_title"] in titles]
    done = {r["requested_title"] for r in rows}
    total = len(titles)
    titles = [title for title in titles if title not in done]
    for start in range(0, len(titles), 10):
        batch = titles[start:start+10]
        params = {
            "action": "query", "format": "json", "redirects": 1,
            "prop": "extracts|info|revisions|pageprops", "explaintext": 1, "exintro": 1,
            "exlimit": 10, "inprop": "url", "rvprop": "ids|timestamp",
            "titles": "|".join(batch), "maxlag": 5,
        }
        for attempt in range(4):
            response = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=45)
            if response.status_code != 429:
                break
            print("Wikipedia requested slower access; waiting before retry", flush=True)
            time.sleep(min(45, 15 * (attempt + 1)))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        query = payload["query"]
        aliases = {x["from"]: x["to"] for field in ("normalized", "redirects")
                   for x in query.get(field, [])}
        pages = {p["title"]: p for p in query["pages"].values()}
        for requested in batch:
            title, visited = requested, set()
            while title in aliases and title not in visited:
                visited.add(title)
                title = aliases[title]
            page = pages.get(title, {})
            extract = page.get("extract", "")
            revision = next(iter(page.get("revisions", [])), {})
            row = {"requested_title": requested, "title": title,
                   "url": page.get("fullurl"), "page_id": page.get("pageid"),
                   "revision_id": revision.get("revid"),
                   "revision_timestamp": revision.get("timestamp"),
                   "retrieved_at": datetime.now(timezone.utc).isoformat(),
                   "missing": "missing" in page or not page,
                   "disambiguation": "disambiguation" in page.get("pageprops", {}),
                   "license": "CC BY-SA (verify article footer and attribution for reuse)",
                   "license_url": "https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
                   "attribution": "Wikipedia contributors",
                   "text_sha256": hashlib.sha256(extract.encode()).hexdigest(),
                   "text": extract, "status": "research_snapshot_not_training_approved"}
            if revision.get("revid"):
                row["permalink"] = "https://en.wikipedia.org/w/index.php?oldid=" + str(revision["revid"])
            rows.append(row)
        # Save after every batch so an interrupted read can be inspected.
        (args.output / "wikipedia.private.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        print(f"Fetched {len(rows)}/{total} concept titles", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
