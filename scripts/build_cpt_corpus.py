"""Build a versioned, source-traceable CPT release; unknown rights fail closed.

The private build includes staging, quarantine and split Parquet files. Only
aggregate evidence belongs in Git. No model calls or semantic rewrites occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from tokenizers import Tokenizer

VERSION = 'cpt-corpus-v1.0.2'
TOKENIZER_SHA = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
NOISE = {'Edit', 'Copy page', 'Copy page as Markdown for LLMs', 'Open in ChatGPT',
         'Open in Claude', 'Ask ChatGPT about this page', 'Ask Claude about this page',
         'Download this page as a PDF', 'Show Contents', 'Hide Contents'}
BENCHMARK_SOURCES = {'mit_key', 'ports_pdf', 'ports_book', 'ivanov',
                     'international_business', 'transport_glossary5', 'glossary_itf'}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    # NFC deliberately preserves superscripts, subscripts and mathematical signs.
    text = unicodedata.normalize('NFC', text).replace('\u00ad', '').replace('\xa0', ' ')
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', text)
    return '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()).strip()


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def shingles(text, n=5):
    ws = words(text)
    return {' '.join(ws[i:i+n]) for i in range(max(0,len(ws)-n+1))}


def overlap(text, fingerprints):
    ws = words(text)
    hits = {sha(' '.join(ws[i:i+13])) for i in range(max(0,len(ws)-12))}
    return len(hits & fingerprints['ngrams']) or int(sha(' '.join(ws)) in fingerprints['exact'])


def rights(row):
    sid = row['id']
    if sid in BENCHMARK_SOURCES:
        return 'benchmark_source_quarantine'
    url=urlsplit(row['url'])
    if sid.startswith('erp_') and url.hostname=='docs.frappe.io' and url.path.startswith('/erpnext/'):
        return 'release_cc_by_sa'
    if sid.startswith('mit_') or sid in {'openstax_is','openstax_scm','risk_scm','intro_logistics'}:
        return 'noncommercial_research_only'
    if sid in {'frappe_license','mit_terms','inventory_license','china_terms','warehouse_portal','gs1_guidelines'}:
        return 'metadata_only'
    return 'rights_review_required'


def table_text(node):
    if node.select('[rowspan], [colspan]'):
        return node.get_text(' ',strip=True), ['complex_table_requires_review']
    rows = [[normalize(c.get_text(' ',strip=True)).replace('|','\\|')
             for c in tr.find_all(['td','th'],recursive=False)] for tr in node.find_all('tr')]
    rows = [r for r in rows if r]
    if not rows or len({len(r) for r in rows}) != 1:
        return node.get_text(' ',strip=True), ['irregular_table_requires_review']
    # A neutral TSV preserves rows without inventing a header where none exists.
    return '```text\n'+'\n'.join('\t'.join(r) for r in rows)+'\n```', []


def html_blocks(raw, row):
    soup = BeautifulSoup(raw,'lxml')
    title = normalize(soup.title.get_text(' ',strip=True)) if soup.title else row['title']
    if row['id']=='scm_vscht':
        # Parse JSON literals only. Never execute page scripts.
        source = raw.decode('utf-8',errors='replace')
        marker = 'var dataComponents = '
        data,_ = json.JSONDecoder().raw_decode(source.split(marker,1)[1])
        blocks=[]
        for component in data:
            content = component.get('content',{})
            if component.get('deleted') or content.get('delete') or content.get('deleted'):
                continue
            if content.get('text'):
                s=BeautifulSoup(content['text'],'lxml')
                text=normalize(s.get_text(' ',strip=True))
                if text:blocks.append(dict(text=text,section=[],locator=component.get('id'),
                    flags=['embedded_book_order_and_third_party_rights_review']))
        return title,blocks,['embedded_book_review_required']
    if row['id'].startswith('erp_'):
        root=soup.select_one('#wiki-content')
        if root is None:
            return title,[],['expected_article_missing']
    else:
        root=soup.select_one('article .field--name-body') or soup.select_one('article') or soup.select_one('main')
        if root is None:
            return title,[],['reliable_body_selector_missing']
    for node in root.select('script,style,nav,header,footer,button,form,svg,iframe'):
        node.decompose()
    heading=[];blocks=[]
    selected=root.find_all(['h1','h2','h3','h4','h5','h6','p','li','pre','table'])
    for i,node in enumerate(selected):
        if node.find_parent(['table','pre']) is not None:
            continue
        if node.name=='p' and node.find_parent('li') is not None:
            continue
        if node.name=='li' and node.find_parent('li') is not None:
            continue
        text=normalize(node.get_text(' ',strip=True))
        if node.name.startswith('h'):
            level=int(node.name[1])
            while heading and heading[-1][0]>=level:heading.pop()
            heading.append((level,text))
            continue
        if not text or text in NOISE:
            continue
        flags=[]
        if any(re.fullmatch(r'(?i)(?:\d+[. ]*)?related topics', h[1]) for h in heading):
            flags.append('related_topics_navigation')
        if node.name=='table':text,flags=table_text(node)
        elif node.name=='pre':text='```\n'+unicodedata.normalize('NFC',node.get_text()).strip('\n')+'\n```'
        elif node.name=='li':text='- '+text
        if node.select('math'):
            flags.append('mathml_requires_review')
        if '\ufffd' in text:flags.append('unicode_replacement_character')
        if re.search(r'(?i)(as (?:shown|illustrated|seen)|see (?:the )?(?:image|figure|diagram)|following (?:image|diagram|screenshot))',text):
            flags.append('visual_dependency_requires_review')
        if re.search(r'(?i)(api[_ -]?key|password|secret)\s*[:=]\s*[\"\x27]?[A-Za-z0-9_/-]{20,}',text):
            flags.append('credential_like_example_requires_review')
        if re.search(r'<\|(?:im_start|im_end|endoftext)\|>',text):
            flags.append('model_control_token')
        blocks.append(dict(text=text,section=[h[1] for h in heading],locator=f'body:block:{i}',flags=flags))
    return title,blocks,[]


def pdf_blocks(row, extraction_root):
    textfile=extraction_root/(row['sha256']+'.txt')
    if not textfile.exists():
        return row['title'],[],['pdf_extraction_missing']
    pages=textfile.read_text(encoding='utf-8').split('\f')
    edges=Counter()
    for p in pages:
        lines=[normalize(x) for x in p.splitlines() if x.strip()]
        edges.update(set(lines[:2]+lines[-2:]))
    recurring={x for x,n in edges.items() if n>=max(3,len(pages)*.3) and len(x)<160}
    blocks=[]
    for page,p in enumerate(pages,1):
        lines=[normalize(x) for x in p.splitlines()]
        lines=[x for x in lines if x not in recurring and not re.fullmatch(r'\d{1,4}',x)]
        text=normalize('\n'.join(lines))
        if text:
            flags=['pdf_layout_formula_review_pending']
            if len(text)<50:flags.append('low_text_page')
            blocks.append(dict(text=text,section=[f'Page {page}'],locator=f'pdf:page:{page}',flags=flags))
    return row['title'],blocks,[]


def make_chunks(blocks, title, tokenizer, limit=4095):
    result=[];current=[]
    def render(items):
        prefix='ERPNext documentation: '+title+'\n'
        parts=[];previous=None
        for item in items:
            if item['section']!=previous and item['section']:
                parts.append(' > '.join(item['section']))
            parts.append(item['text']);previous=item['section']
        return prefix+'\n'+'\n\n'.join(parts)
    def emit(items):
        if not items:return
        text=render(items)
        result.append(dict(text=text,block_ids=[b['block_id'] for b in items],
                           section=items[0]['section'],content_tokens=len(tokenizer.encode(text,add_special_tokens=False).ids)))
    for block in blocks:
        if len(tokenizer.encode(render([block]),add_special_tokens=False).ids)>limit:
            emit(current);current=[]
            # Do not split tables, code, or paragraphs mid-sentence just to fill a batch.
            raise ValueError('Oversized indivisible block; requires source review')
        if len(tokenizer.encode(render(current+[block]),add_special_tokens=False).ids)>limit:
            emit(current);current=[]
        current.append(block)
    emit(current)
    return result


def dump_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--fingerprints',type=Path,required=True)
    p.add_argument('--archive-root',type=Path,default=Path('CPT_resources'))
    p.add_argument('--previous-build',type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Build directories are immutable; use a new version')
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()==TOKENIZER_SHA
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding();tokenizer.no_truncation()
    assert tokenizer.token_to_id('<|endoftext|>') is not None
    fp=json.loads(gzip.decompress(args.fingerprints.read_bytes()))
    assert fp['cases']==1672 and fp['raw_text_included'] is False
    assert fp['source_sha256']=='b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
    fp['ngrams']=set(fp['ngrams']);fp['exact']=set(fp['exact'])
    args.out.mkdir(parents=True)
    records=[]
    for manifest in sorted(args.archive_root.glob('material_survey_*/manifest.jsonl')):
        records.extend(json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines())
    sources=[];documents=[];quarantine=[];seen=set();admitted=[]
    extraction=args.archive_root/'material_survey_20260910_index/text_for_review'
    for row in sorted(records,key=lambda r:(r['id'],r['url'])):
        sid=sha(row['url']);policy=rights(row)
        source=dict(source_id=sid,url=row['url'],collection_id=row['id'],rights=policy,
            raw_sha256=row.get('sha256'),retrieved_at=row.get('fetched_at'),status=row['status'])
        sources.append(source)
        if row['status']!='downloaded':continue
        if row['sha256'] in seen:continue
        seen.add(row['sha256'])
        raw=Path(row['local_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        try:
            if row['format']=='pdf':title,blocks,flags=pdf_blocks(row,extraction)
            else:title,blocks,flags=html_blocks(raw,row)
        except Exception as exc:
            title,blocks,flags=row['title'],[],['extraction_error:'+type(exc).__name__]
        for i,block in enumerate(blocks):
            block['block_id']=sha(sid+'|'+str(i)+'|'+block['text'])
            block['text_sha256']=sha(block['text'])
        family='erpnext:'+normalize(title).casefold() if row['id'].startswith('erp_') else sid
        doc=dict(source_id=sid,source_url=row['url'],raw_sha256=row['sha256'],title=title,
            family_id=sha(family),rights=policy,blocks=blocks,flags=flags,
            cleaned_sha256=sha('\n\n'.join(b['text'] for b in blocks)))
        documents.append(doc)
        if policy!='release_cc_by_sa' or flags or not blocks:
            quarantine.append(dict(source_id=sid,scope='document',reasons=[policy]+flags))
        else:admitted.append(doc)
    # Exact and near duplicate document families are resolved before splitting.
    representatives=[];duplicate_docs={}
    for doc in sorted(admitted,key=lambda d:d['source_id']):
        sig=shingles('\n'.join(b['text'] for b in doc['blocks']))
        match=None
        for other,other_sig in representatives:
            similarity=len(sig&other_sig)/max(1,len(sig|other_sig))
            facts=lambda d:re.findall(r'\d+(?:\.\d+)?|\b(?:not|never|unless|except|must|only|cannot)\b',' '.join(b['text'] for b in d['blocks']).lower())
            if doc['cleaned_sha256']==other['cleaned_sha256'] or (similarity>=.85 and facts(doc)==facts(other)):
                match=other;break
        if match:
            duplicate_docs[doc['source_id']]=match['source_id']
            quarantine.append(dict(source_id=doc['source_id'],scope='document',reasons=['duplicate_document'],canonical_source_id=match['source_id']))
        else:representatives.append((doc,sig))
    # Stable family hash split. A family never changes split with new additions.
    split={d['family_id']:('validation' if int(d['family_id'][:8],16)%10==0 else 'train') for d,_ in representatives}
    data={'train':[],'validation':[]};owned={}
    for doc,_ in representatives:
        accepted=[]
        sections=[]
        for block in doc['blocks']:
            if not sections or sections[-1][0]!=block['section']:
                sections.append((block['section'],[]))
            sections[-1][1].append(block)
        for section,section_blocks in sections:
            text='\n\n'.join(b['text'] for b in section_blocks)
            reasons=sorted({flag for b in section_blocks for flag in b['flags']})
            if len(words(text))<4:reasons.append('fragment_too_short')
            hits=overlap(text,fp)
            if hits:reasons.append('benchmark_13gram_or_exact_match')
            # Never drop an isolated condition from a retained section: keep or
            # quarantine the entire section as one semantic unit.
            key=sha(' '.join(words(text)))
            if key in owned:reasons.append('duplicate_section')
            if reasons:
                quarantine.append(dict(source_id=doc['source_id'],block_ids=[b['block_id'] for b in section_blocks],scope='section',reasons=reasons,benchmark_matches=hits))
                continue
            owned[key]=doc['source_id'];accepted.extend(section_blocks)
        try:chunks=make_chunks(accepted,doc['title'],tokenizer)
        except ValueError as exc:
            quarantine.append(dict(source_id=doc['source_id'],scope='document',reasons=[str(exc)]));continue
        for chunk in chunks:
            if overlap(chunk['text'],fp):
                quarantine.append(dict(source_id=doc['source_id'],scope='chunk',reasons=['benchmark_match_across_blocks']));continue
            chunk.update(id=sha(doc['source_id']+'|'+chunk['text']),source_id=doc['source_id'],
                source_url=doc['source_url'],raw_sha256=doc['raw_sha256'],family_id=doc['family_id'],
                license='CC BY-SA',attribution='Frappe Technologies Pvt. Ltd. and documentation contributors',
                license_evidence='https://docs.frappe.io/legal/others/license-and-trademark',
                processor_version=VERSION,tokenizer_sha256=TOKENIZER_SHA,
                sequence_tokens=chunk['content_tokens']+1,token_count=chunk['content_tokens']+1,
                text_sha256=sha(chunk['text']),eos_token_id=248046,eos_added_by_loader=True)
            assert chunk['sequence_tokens']<=4096
            data[split[doc['family_id']]].append(chunk)
    assert data['train'] and data['validation'], 'Need nonempty document-disjoint splits'
    assert not {x['family_id'] for x in data['train']}&{x['family_id'] for x in data['validation']}
    assert len({x['id'] for group in data.values() for x in group})==sum(map(len,data.values()))
    dump_jsonl(args.out/'sources.jsonl',sources)
    dump_jsonl(args.out/'staging_documents.jsonl',documents)
    dump_jsonl(args.out/'quarantine.jsonl',quarantine)
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name,rows in data.items():
        dump_jsonl(args.out/(name+'.jsonl'),rows)
        pq.write_table(pa.Table.from_pylist(rows),args.out/(name+'.parquet'),compression='zstd')
    report=dict(version=VERSION,input_urls=len(records),downloaded_unique_sources=len(documents),
        rights_counts=dict(Counter(d['rights'] for d in documents)),
        extracted_blocks=sum(len(d['blocks']) for d in documents),admitted_documents=len(admitted),
        duplicate_documents=len(duplicate_docs),quarantine_reasons=dict(Counter(reason for q in quarantine for reason in q['reasons'])),
        splits={name:dict(records=len(rows),documents=len({r['source_id'] for r in rows}),
            content_tokens=sum(r['content_tokens'] for r in rows),sequence_tokens=sum(r['sequence_tokens'] for r in rows)) for name,rows in data.items()},
        tokenizer_sha256=TOKENIZER_SHA,benchmark_source_sha256=fp['source_sha256'],
        decontamination='13-word exact windows and >=8-word whole-block matches; not semantic contamination proof',
        validation_scope='document-disjoint ERPNext NLL development split, not independent domain benchmark',
        rewriting=False,training_started=False,release_license='CC BY-SA; keep attribution and ShareAlike obligations',
        source_split_assignments=split)
    report['build_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['fingerprint_file_sha256']=hashlib.sha256(args.fingerprints.read_bytes()).hexdigest()
    if args.previous_build:
        previous=json.loads((args.previous_build/'manifest.safe.json').read_text(encoding='utf-8'))
        for family,value in previous['source_split_assignments'].items():
            if family in split:assert split[family]==value
        old={r['id'] for name in ('train','validation') for r in (json.loads(x) for x in (args.previous_build/(name+'.jsonl')).read_text(encoding='utf-8').splitlines())}
        new={r['id'] for rows in data.values() for r in rows}
        report['incremental_delta']=dict(added=len(new-old),removed=len(old-new),unchanged=len(new&old),existing_family_splits_unchanged=True,
            previous_manifest_sha256=hashlib.sha256((args.previous_build/'manifest.safe.json').read_bytes()).hexdigest())
    (args.out/'ATTRIBUTION.md').write_text('# Attribution and corpus use\n\nReleased text: ERPNext documentation by Frappe Technologies Pvt. Ltd. and documentation contributors.\n\nLicense: Creative Commons Attribution-ShareAlike (CC BY-SA); the publisher policy does not specify a version. https://docs.frappe.io/legal/others/license-and-trademark\n\nChanges: article extraction, removal of navigation/media, paragraph filtering, deduplication and chunking. Preserve source URLs, notices and ShareAlike obligations in downstream use. Trademarks, logos and illustrations are excluded.\n\nStaging and quarantine have mixed rights and MUST NOT be treated as the released training set.\n',encoding='utf-8')
    report['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out/'manifest.safe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
