"""Index candidate archives and extract PDF text for inspection, not model training."""
from collections import Counter
import json
import logging
from pathlib import Path
import re

from pypdf import PdfReader


def main():
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    base = Path('CPT_resources')
    output = base/'material_survey_20260910_index'
    output.mkdir(exist_ok=True)
    texts = output/'text_for_review'
    texts.mkdir(exist_ok=True)
    rows = []
    for manifest in sorted(base.glob('material_survey_20260910*/manifest.jsonl')):
        rows.extend(json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines())
    unique = {}
    for row in rows:
        if row['status'] == 'downloaded':
            unique.setdefault(row['sha256'], row)
    checks = []
    for sha, row in unique.items():
        if row['format'] != 'pdf':
            continue
        try:
            pdf = PdfReader(row['local_path'])
            pages = [p.extract_text() or '' for p in pdf.pages]
            (texts/(sha+'.txt')).write_text('\n\f\n'.join(pages), encoding='utf-8')
            checks.append(dict(sha256=sha, id=row['id'], pages=len(pages), readable=True,
                text_chars=sum(map(len,pages)), words=sum(len(re.findall(r'\S+',p)) for p in pages),
                low_text_pages=sum(len(p.strip())<50 for p in pages), formula_layout_qa='pending'))
        except Exception as exc:
            checks.append(dict(sha256=sha, id=row['id'], readable=False, error=type(exc).__name__))
    summary = dict(attempted=len(rows), downloaded=sum(r['status']=='downloaded' for r in rows),
        failed=sum(r['status']=='failed' for r in rows), unique_files=len(unique),
        pdf_files=sum(r['format']=='pdf' for r in unique.values()),
        html_files=sum(r['format']=='html' for r in unique.values()),
        unique_bytes=sum(r['bytes'] for r in unique.values()),
        pdf_pages=sum(r.get('pages',0) for r in checks),
        pdf_text_words=sum(r.get('words',0) for r in checks),
        downloaded_by_source=dict(Counter(r['id'] for r in rows if r['status']=='downloaded')),
        training_approved=0, full_text_near_dedup='pending', benchmark_overlap_scan='pending',
        note='Counts include resource landing pages; not a count of books or approved training examples.')
    Path('docs/cpt_material_collection_20260910.safe.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    Path('docs/cpt_material_pdf_readability_20260910.safe.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    safe = [{k:v for k,v in r.items() if k!='local_path'} for r in rows]
    Path('docs/cpt_material_download_manifest_20260910.safe.json').write_text(json.dumps(safe,indent=2),encoding='utf-8')
    lines = ['# CPT candidate material archive', '',
        'Research collection only. Rights, contamination, extraction and relevance checks precede training.', '',
        '| Source | File | URL | Rights |', '|---|---|---|---|']
    for r in unique.values():
        path=Path(r['local_path']).resolve().as_posix()
        lines.append(f"| {r['id']} | [{r['format']}]({path}) | [source]({r['url']}) | {r.get('rights_status','review required')} |")
    (output/'INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
