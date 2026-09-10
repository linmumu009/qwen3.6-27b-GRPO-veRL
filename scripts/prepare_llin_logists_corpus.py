"""Local, provenance-preserving preparation of user-supplied logistics books.

No uploads, generated facts, automatic training, or assertion of redistribution rights.
Known benchmark source families and uncertain extraction blocks stay outside train.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import fitz
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
import pyarrow as pa
import pyarrow.parquet as pq

VERSION='llin-logists-v4'
TOKENIZER_SHA='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
# These are source-family exclusions, not merely literal answer matches.
BENCHMARK_PREFIXES=('43_Book_', 'MITx_', 'KS-GQ-', 'challenges-and-')
STARTS={'43_Book_':24,'Beyond Lean':11,'Container terminals':15,'Distribution logistics':12,
        'Fundamentals':4,'International Trade':3,'JSI_':11,'Knoop':14,'KS-GQ-':11,
        'Managing supply':18,'MITx_':19,'wh-sci':27}
MATH_FONT=re.compile(r'(?i)(cmsy|cmmi|cmex|math|symbol|msam|msbm)')
# Article opening pages checked against this source's extracted headings.
DISTRIBUTION_ARTICLES=(15,39,59,77,99,117,133,153,171,197,215,233,255)


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def norm(text):
    return unicodedata.normalize('NFC',text).replace('\u00ad','').replace('\xa0',' ').replace('ﬁ','fi').replace('ﬂ','fl')


def ws(text):return re.findall(r'[a-z0-9]+',text.lower())


class NearDuplicates:
    """Exact 5-word-shingle Jaccard screening for prose blocks of >=40 words."""
    def __init__(self):
        self.postings={};self.sizes=[]

    def add(self,text):
        words=ws(text)
        if len(words)<40:return False
        shingles={' '.join(words[i:i+5]) for i in range(len(words)-4)}
        shared=Counter(i for s in shingles for i in self.postings.get(s,()))
        if any(n/(len(shingles)+self.sizes[i]-n)>=.85 for i,n in shared.items()):return True
        index=len(self.sizes);self.sizes.append(len(shingles))
        for s in shingles:self.postings.setdefault(s,[]).append(index)
        return False


def fingerprint_hit(text,fp):
    words=ws(text)
    return bool({sha(' '.join(words[i:i+13])) for i in range(max(0,len(words)-12))}&fp['ngrams']) or sha(' '.join(words)) in fp['exact']


def join_lines(lines,vocabulary):
    text=''
    for line in lines:
        line=re.sub(r'[ \t]+',' ',norm(line)).strip()
        if not line:continue
        if text.endswith('-') and re.match(r'^[a-z]',line):
            end=re.search(r'([A-Za-z]+)-$',text);start=re.match(r'([A-Za-z]+)',line)
            if end and start and (end[1]+start[1]).lower() in vocabulary:
                text=text[:-1]+line
            else:text+=line  # Keep a possible semantic compound hyphen.
        else:text+=(' ' if text else '')+line
    return text


def read_pdf(path,source_id):
    doc=fitz.open(path)
    raw_text=[page.get_text() for page in doc]
    vocabulary={w for w,n in Counter(re.findall(r'\b[a-z]{4,}\b',norm('\n'.join(raw_text)).lower())).items() if n>=2}
    page_data=[];edge_counts=Counter()
    for page in doc:
        blocks=[b for b in page.get_text('dict')['blocks'] if b.get('type')==0]
        blocks.sort(key=lambda b:(round(b['bbox'][1],1),b['bbox'][0]))
        keys=[]
        for b in blocks:
            text=' '.join(s['text'] for l in b['lines'] for s in l['spans'])
            if b['bbox'][1]<page.rect.height*.17 or b['bbox'][3]>page.rect.height*.88:
                key=re.sub(r'\d+','#',norm(text)).strip()
                if len(key)<220:keys.append(key)
        edge_counts.update(set(keys));page_data.append(blocks)
    recurring={k for k,n in edge_counts.items() if n>=max(3,len(doc)*.015)}
    start=next((v for k,v in STARTS.items() if path.name.startswith(k)),1)
    toc=[(level,title,page) for level,title,page in doc.get_toc() if not re.search(r'(?i)(download pdf|fulltext|front-matter|cover)',title)]
    result=[];section=[];chapter='book';toc_index=0;backmatter=False
    for number,(page,blocks) in enumerate(zip(doc,page_data),1):
        while toc_index<len(toc) and toc[toc_index][2]<=number:
            level,title,_=toc[toc_index]
            while section and section[-1][0]>=level:section.pop()
            section.append((level,norm(title)))
            toc_index+=1
        if section:
            chapter=' / '.join(x[1] for x in section[:2])
        for ordinal,b in enumerate(blocks):
            lines=[''.join(s['text'] for s in line['spans']) for line in b['lines']]
            spans=[s for l in b['lines'] for s in l['spans']]
            text=join_lines(lines,vocabulary)
            if not text:continue
            flags=[];key=re.sub(r'\d+','#',norm(' '.join(s['text'] for s in spans))).strip()
            edge=b['bbox'][1]<page.rect.height*.17 or b['bbox'][3]>page.rect.height*.88
            if number<start:flags.append('front_matter')
            if edge and (key in recurring or re.fullmatch(r'[\divxIVX .—–-]+',text)):flags.append('running_header_footer')
            if re.search(r'(?:\.\s*){5,}',text):flags.append('table_of_contents')
            if any('GlyphLessFont' in s['font'] for s in spans):flags.append('scanned_ocr_requires_review')
            if any(MATH_FONT.search(s['font']) for s in spans) or re.search(r'[∑∫∏√≤≥≈∂∞]',text):flags.append('formula_requires_review')
            if '\ufffd' in text or re.search(r'\(cid:\d+\)',text):flags.append('encoding_damage')
            if re.search(r'<\|(?:im_start|im_end|endoftext)\|>',text):flags.append('model_control_token')
            # Independent text runs on the same baseline are often table cells or diagrams.
            line_boxes=[l['bbox'] for l in b['lines']]
            parallel=any(abs(a[1]-c[1])<3 and (a[2]+8<c[0] or c[2]+8<a[0]) for i,a in enumerate(line_boxes) for c in line_boxes[i+1:])
            if parallel:flags.append('tabular_or_diagram_layout')
            if re.search(r'(?i)\b(?:figure|fig\.|table|equation)\s*[\d(]|as (?:shown|illustrated)|following (?:diagram|image)',text):flags.append('visual_or_equation_reference')
            if re.search(r'(?i)\b(?:Begin|End)\b',text) and re.search(r'(?i)\b(?:Process|Define|Tabulate|Decrement)\b',text):flags.append('pseudocode_requires_layout_review')
            if len(ws(text))<8:flags.append('short_fragment')
            if len(ws(text))>15 and sum(len(w)==1 for w in ws(text))/len(ws(text))>.22:flags.append('fragmented_text')
            if re.match(r'(?i)^(?:references|bibliography|index|answers to|solutions to)\s*$',text):backmatter=True
            if re.match(r'(?i)^(?:chapter\s+\d+|\d+\s+[A-Z])',text) and len(text)<160:backmatter=False
            if backmatter:flags.append('references_or_answers')
            result.append(dict(block_id=sha(f'{source_id}:{number}:{ordinal}:{text}'),text=text,text_sha256=sha(text),
                page=number,bbox=[round(x,2) for x in b['bbox']],section=[x[1] for x in section],
                group_id=sha(source_id+'|'+chapter),flags=sorted(set(flags))))
    return result,len(doc)


def read_zip(path,source_id):
    result=[]
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(),key=lambda i:i.filename):
            # Read members in memory; never execute HTML or extract arbitrary paths.
            if not re.search(r'/s\d+-\d+[^/]*\.html$',info.filename):continue
            if info.file_size>20_000_000:raise ValueError('Unexpectedly large HTML member')
            soup=BeautifulSoup(archive.read(info),'lxml')
            for n in soup.select('script,style,nav,.navbar,.copyright'):n.decompose()
            root=soup.select_one('.section') or soup.select_one('#book-content') or soup.body
            if root is None:continue
            for index,node in enumerate(root.find_all(['p','li','h1','h2','h3'])):
                if node.find_parent('li') is not None:continue
                text=re.sub(r'\s+',' ',norm(node.get_text(' ',strip=True))).strip()
                if not text:continue
                result.append(dict(block_id=sha(f'{source_id}:{info.filename}:{index}:{text}'),text=text,text_sha256=sha(text),
                    page=0,member=info.filename,bbox=[],section=[info.filename],group_id=sha(source_id+'|'+info.filename.split('-')[0]),flags=[]))
    return result,0


def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def build(args):
    if args.out.exists():raise FileExistsError('Immutable build: choose a new output directory')
    assert sha(args.tokenizer.read_bytes())==TOKENIZER_SHA
    tokenizer=Tokenizer.from_file(str(args.tokenizer));tokenizer.no_padding();tokenizer.no_truncation()
    fp=json.loads(gzip.decompress(args.fingerprints.read_bytes()))
    assert fp['cases']==1672 and fp['raw_text_included'] is False
    fp['ngrams']=set(fp['ngrams']);fp['exact']=set(fp['exact'])
    args.out.mkdir(parents=True);sources=[];staging=[];quarantine=[];released=[];seen_sources=set();owned={};near=NearDuplicates()
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in ('.pdf','.zip'):continue
        source_id=sha(path.read_bytes())
        if source_id in seen_sources:continue
        seen_sources.add(source_id)
        from read_llin_mineru import read_mineru
        mineru_parts=json.loads(args.parts.read_text(encoding='utf-8')) if args.parts and args.parts.exists() else []
        mineru=read_mineru(path.name,source_id,mineru_parts,args.mineru_dir) if args.mineru_dir else None
        cache=args.cache/(source_id+'.json.gz');cache.parent.mkdir(parents=True,exist_ok=True)
        if mineru:
            blocks,pages=mineru
            with fitz.open(path) as original:
                assert pages==len(original), 'MinerU parts must cover every original page'
                if path.name.startswith('Knoop'):
                    chapters=[(title,page) for level,title,page in original.get_toc() if level==3]
                    for b in blocks:
                        prior=[title for title,page in chapters if page<=b['page']]
                        if prior:
                            b['group_id']=sha(source_id+'|'+prior[-1]);b['section']=[prior[-1]]
            vocabulary={w for w,n in Counter(re.findall(r'\b[a-z]{4,}\b','\n'.join(b['text'] for b in blocks).lower())).items() if n>=2}
            for b in blocks:
                b['text']=re.sub(r'([A-Za-z]+)-\n([a-z]+)',lambda m:m[1]+m[2] if (m[1]+m[2]).lower() in vocabulary else m[1]+'-'+m[2],b['text'])
                b['text_sha256']=sha(b['text'])
            if path.name.startswith('Distribution logistics'):
                article_titles={page:next(b['text'].lstrip('# ') for b in blocks if b['page']==page and b['kind']=='title') for page in DISTRIBUTION_ARTICLES}
                for b in blocks:
                    prior=[page for page in DISTRIBUTION_ARTICLES if page<=b['page']]
                    if prior:
                        b['group_id']=sha(source_id+'|article|'+str(prior[-1]))
                        b['section']=[article_titles[prior[-1]]]+b['section'][-1:]
                    if b['page']>=295:b['flags'].append('publisher_back_matter')
        elif cache.exists():parsed=json.loads(gzip.decompress(cache.read_bytes()));blocks=parsed['blocks'];pages=parsed['pages']
        else:
            blocks,pages=read_pdf(path,source_id) if path.suffix.lower()=='.pdf' else read_zip(path,source_id)
            cache.write_bytes(gzip.compress(json.dumps(dict(blocks=blocks,pages=pages)).encode(),mtime=0))
        contaminated_family=path.name.startswith(BENCHMARK_PREFIXES)
        sources.append(dict(source_id=source_id,filename=path.name,bytes=path.stat().st_size,pages=pages,
            benchmark_source_family=contaminated_family,provenance='user_supplied',rights='original source terms retained; no new redistribution grant',blocks=len(blocks),extractor='MinerU' if mineru else 'local text layer'))
        staging.extend(dict(source_id=source_id,**b) for b in blocks)
        # A rejected block is a hard boundary: do not glue its preceding premise to later content.
        current=[];current_group=None;source_records=[]
        title=path.stem.split(' (')[0].replace('_',' ')
        def render(items):
            heading=' > '.join(items[0].get('section',[])) if items else ''
            return 'Source: '+title+('\nSection: '+heading if heading else '')+'\n\n'+'\n\n'.join(b['text'] for b in items)
        def emit():
            nonlocal current
            if not current:return
            text=render(current)
            if len(ws('\n'.join(b['text'] for b in current)))<40:
                quarantine.append(dict(source_id=source_id,block_ids=[b['block_id'] for b in current],reasons=['short_context']))
            elif fingerprint_hit(text,fp):
                quarantine.append(dict(source_id=source_id,block_ids=[b['block_id'] for b in current],reasons=['benchmark_overlap_across_blocks']))
            else:
                n=len(tokenizer.encode(text,add_special_tokens=False).ids)
                source_records.append(dict(id=sha(source_id+'|'+text),text=text,text_sha256=sha(text),source_id=source_id,
                    source_file=path.name,source_pages=sorted({page for b in current for page in [b['page']]+b.get('additional_pages',[])}),group_id=current[0]['group_id'],
                    block_ids=[b['block_id'] for b in current],content_tokens=n,token_count=n+1,eos_token_id=248046,
                    processor_version=VERSION,tokenizer_sha256=TOKENIZER_SHA))
            current=[]
        for b in blocks:
            flags=b['flags'].copy()
            if contaminated_family:flags.append('benchmark_source_family')
            if fingerprint_hit(b['text'],fp):flags.append('benchmark_exact_overlap')
            key=sha(' '.join(ws(b['text'])))
            if key in owned:flags.append('duplicate_block')
            if len(tokenizer.encode(render([b]),add_special_tokens=False).ids)>4095:flags.append('oversized_block')
            if not flags and near.add(b['text']):flags.append('near_duplicate_block')
            if flags:
                if not set(flags)<={'running_header_footer','page_furniture'}:emit()
                quarantine.append(dict(source_id=source_id,block_ids=[b['block_id']],reasons=sorted(set(flags))));continue
            owned[key]=b['block_id']
            if current and (current_group!=b['group_id'] or len(tokenizer.encode(render(current+[b]),add_special_tokens=False).ids)>4095):emit()
            current_group=b['group_id'];current.append(b)
        emit();released.extend(source_records)
        print(json.dumps(dict(source=path.name,pages=pages,blocks=len(blocks),records=len(source_records))),flush=True)
    # Split whole chapter/book groups. Never randomly split adjacent snippets.
    groups=sorted({r['group_id'] for r in released})
    split={g:('validation' if int(g[:8],16)%10==0 else 'train') for g in groups}
    data={s:[r for r in released if split[r['group_id']]==s] for s in ('train','validation')}
    assert all(data.values())
    write_jsonl(args.out/'sources.jsonl',sources);write_jsonl(args.out/'staging.jsonl',staging);write_jsonl(args.out/'quarantine.jsonl',quarantine)
    for name,rows in data.items():
        write_jsonl(args.out/(name+'.jsonl'),rows)
        pq.write_table(pa.Table.from_pylist(rows),args.out/(name+'.parquet'),compression='zstd')
    manifest=dict(version=VERSION,sources=sources,pages=sum(s['pages'] for s in sources),staging_blocks=len(staging),
        quarantine_reasons=dict(Counter(f for r in quarantine for f in r['reasons'])),
        split_groups=split,splits={k:dict(records=len(v),sources=len({r['source_id'] for r in v}),content_tokens=sum(r['content_tokens'] for r in v),sequence_tokens=sum(r['token_count'] for r in v)) for k,v in data.items()},
        tokenizer_sha256=TOKENIZER_SHA,fingerprint_sha256=sha(args.fingerprints.read_bytes()),
        benchmark_source_sha256=fp['source_sha256'],script_sha256=sha(Path(__file__).read_bytes()),
        mineru_reader_sha256=sha(Path(__file__).with_name('read_llin_mineru.py').read_bytes()),
        mineru_exports={p.name:sha(p.read_bytes()) for p in sorted(args.mineru_dir.glob('*.json'))} if args.mineru_dir else {},
        parts_manifest_sha256=sha(args.parts.read_bytes()) if args.mineru_dir else None,
        training_started=False,rights_review='Source terms retained, private user-supplied processing; not a public licensed dataset',
        limitations=['No semantic decontamination guarantee','Uncertain OCR, formulas and layout remain quarantined','Chapter split is development NLL only'])
    manifest['artifacts']={p.name:sha(p.read_bytes()) for p in args.out.iterdir() if p.is_file()}
    (args.out/'manifest.safe.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['splits']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('CPT_resources/CPT-LOGISTS'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('CPT_resources/llin-logists-cache-v1'))
    p.add_argument('--tokenizer',type=Path,default=Path('CPT_resources/corpus_build_inputs_20260910/tokenizer.json'))
    p.add_argument('--fingerprints',type=Path,default=Path('CPT_resources/corpus_build_inputs_20260910/benchmark_fingerprints.json.gz'))
    p.add_argument('--mineru-dir',type=Path)
    p.add_argument('--parts',type=Path,default=Path('CPT_resources/llin-mineru-input/parts_manifest.json'))
    build(p.parse_args())
