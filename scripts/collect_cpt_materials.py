"""Archive public candidate sources with provenance; never emit a training dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlsplit, urldefrag
from urllib.request import Request, urlopen


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def fetch(item, root):
    record = dict(item, fetched_at=datetime.now(timezone.utc).isoformat(), training_approved=False)
    try:
        with urlopen(Request(item['url'], headers={'User-Agent': 'CPT-material-research/1.0'}), timeout=25) as response:
            chunks, size, started = [], 0, time.monotonic()
            while True:
                if time.monotonic()-started > 120:
                    raise TimeoutError('Download exceeded 120 seconds')
                chunk = response.read1(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 25_000_000:
                    raise ValueError('File exceeds 25 MB collection limit')
                chunks.append(chunk)
            data = b''.join(chunks)
            final = response.url
            mime = response.headers.get('Content-Type', '')
        is_pdf = data.startswith(b'%PDF-')
        if '.pdf' in urlsplit(final).path.lower() and not is_pdf:
            raise ValueError('PDF URL did not return a PDF')
        suffix = '.pdf' if is_pdf else '.html'
        digest = hashlib.sha256(data).hexdigest()
        path = root/'raw'/(digest+suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        record.update(status='downloaded', final_url=final, content_type=mime,
                      bytes=len(data), sha256=digest, local_path=str(path), format=suffix[1:])
        if not is_pdf and item.get('crawl'):
            parser = Links()
            parser.feed(data.decode('utf-8', errors='replace'))
            candidates = []
            for link in parser.links:
                target = urldefrag(urljoin(final, link))[0]
                parsed = urlsplit(target)
                if parsed.scheme != 'https' or parsed.netloc != urlsplit(final).netloc:
                    continue
                lower = parsed.path.lower()
                if any(t in lower for t in ('exam', 'assignment', 'solution', 'problem-set', 'quiz')):
                    continue
                mode = item['crawl']
                accept = (mode == 'mit' and (lower.endswith('.pdf') or '/resources/' in lower))
                accept |= mode == 'log' and lower.startswith('/en/') and not parsed.query
                accept |= mode == 'pdf' and lower.endswith('.pdf')
                if accept and target != item['url']:
                    candidates.append(dict(item, url=target, parent_url=item['url'],
                        crawl='mit' if mode=='mit' and not lower.endswith('.pdf') else None))
            record['_children'] = candidates
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out/'manifest.jsonl'
    if manifest.exists():
        raise FileExistsError('Use a new collection directory to preserve the previous manifest')
    pending = json.loads(a.catalog.read_text(encoding='utf-8'))
    seen = set()
    records = []
    with manifest.open('x', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        while pending and len(seen) < a.limit:
            batch, pending = pending[:16], pending[16:]
            unique = []
            for item in batch:
                if item['url'] not in seen and len(seen) < a.limit:
                    seen.add(item['url']); unique.append(item)
            for record in pool.map(lambda item: fetch(item, a.out), unique):
                pending.extend(record.pop('_children', []))
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
            print(json.dumps(dict(attempted=len(records), downloaded=sum(x['status']=='downloaded' for x in records), queued=len(pending))), flush=True)
    good = [r for r in records if r['status']=='downloaded']
    summary = dict(attempted=len(records), downloaded=len(good), failed=len(records)-len(good),
        unique_files=len({r['sha256'] for r in good}), pdfs=sum(r['format']=='pdf' for r in good),
        total_unique_bytes=sum(p.stat().st_size for p in (a.out/'raw').glob('*')),
        training_approved=0, notes='Research archive only. License, text extraction, relevance, and contamination review required.')
    (a.out/'summary.safe.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
