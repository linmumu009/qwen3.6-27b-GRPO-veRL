"""Independently validate a private logistics CPT release without starting training."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify(root,inputs,sources):
    manifest=json.loads((root/'manifest.safe.json').read_text(encoding='utf-8'))
    for name,expected in manifest['artifacts'].items():
        assert digest((root/name).read_bytes())==expected,(name,'hash mismatch')
    tokenizer=Tokenizer.from_file(str(inputs/'tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    assert digest((inputs/'tokenizer.json').read_bytes())==manifest['tokenizer_sha256']
    fp_path=inputs/'benchmark_fingerprints.json.gz'
    assert digest(fp_path.read_bytes())==manifest['fingerprint_sha256']
    fp=json.loads(gzip.decompress(fp_path.read_bytes()))
    assert fp['cases']==1672 and fp['raw_text_included'] is False
    forbidden_ngrams=set(fp['ngrams']);forbidden_exact=set(fp['exact'])
    special={i for i,t in tokenizer.get_added_tokens_decoder().items() if t.special}
    catalog={s['source_id']:s for s in rows(root/'sources.jsonl')}
    for sid,s in catalog.items():assert digest((sources/s['filename']).read_bytes())==sid
    blocks={b['block_id']:b for b in rows(root/'staging.jsonl')}
    train=rows(root/'train.jsonl');val=rows(root/'validation.jsonl')
    assert {r['group_id'] for r in train}.isdisjoint(r['group_id'] for r in val)
    all_ids=set();all_bodies=set();all_blocks=set();block_split={};source_counts={}
    summaries={}
    for split,data in [('train',train),('validation',val)]:
        assert data==pq.read_table(root/(split+'.parquet')).to_pylist()
        lengths=[]
        for r in data:
            assert r['id'] not in all_ids;all_ids.add(r['id'])
            assert r['text_sha256']==digest(r['text'])
            assert r['id']==digest(r['source_id']+'|'+r['text'])
            assert not catalog[r['source_id']]['benchmark_source_family']
            assert r['source_file']==catalog[r['source_id']]['filename']
            assert r['eos_token_id']==248046 and r['processor_version']==manifest['version']
            ids=tokenizer.encode(r['text'],add_special_tokens=False).ids
            assert not special.intersection(ids)
            assert len(ids)==r['content_tokens'] and len(ids)+1==r['token_count']<=4096
            lengths.append(len(ids)+1)
            words=re.findall('[a-z0-9]+',r['text'].lower())
            assert digest(' '.join(words)) not in forbidden_exact
            assert all(digest(' '.join(words[i:i+13])) not in forbidden_ngrams for i in range(len(words)-12))
            used=[blocks[bid] for bid in r['block_ids']]
            assert all(b['source_id']==r['source_id'] and not b['flags'] and b['group_id']==r['group_id'] for b in used)
            body='\n\n'.join(b['text'] for b in used)
            assert r['text'].endswith('\n\n'+body)
            assert digest(body) not in all_bodies;all_bodies.add(digest(body))
            assert all(digest(b['text'])==b['text_sha256'] for b in used)
            pages=sorted({p for b in used for p in [b['page']]+b.get('additional_pages',[])})
            assert pages==r['source_pages'] and all(1<=p<=catalog[r['source_id']]['pages'] for p in pages)
            for bid in r['block_ids']:
                assert bid not in all_blocks;all_blocks.add(bid);block_split[bid]=split
            source_counts.setdefault(r['source_file'],Counter())[split+'_tokens']+=len(ids)
            source_counts[r['source_file']][split+'_records']+=1
        assert len(data)==manifest['splits'][split]['records']
        assert sum(lengths)==manifest['splits'][split]['sequence_tokens']
        lengths.sort()
        summaries[split]=dict(records=len(data),content_tokens=sum(lengths)-len(data),max_sequence_tokens=max(lengths),
            median_sequence_tokens=lengths[len(lengths)//2],tables=sum('<table>' in r['text'] for r in data),display_formulas=sum('$$' in r['text'] for r in data))
    # Exhaustive shared-shingle candidate counting, independent of the builder's index.
    postings={};sizes=[];identities=[];near_pairs=0
    for bid in sorted(all_blocks):
        words=re.findall('[a-z0-9]+',blocks[bid]['text'].lower())
        if len(words)<40:continue
        shingles={' '.join(words[i:i+5]) for i in range(len(words)-4)}
        overlap=Counter(index for s in shingles for index in postings.get(s,()))
        for index,n in overlap.items():
            if block_split[bid]!=block_split[identities[index]] and n/(len(shingles)+sizes[index]-n)>=.85:near_pairs+=1
        index=len(sizes);sizes.append(len(shingles));identities.append(bid)
        for s in shingles:postings.setdefault(s,[]).append(index)
    assert near_pairs==0
    return dict(status='PASS',manifest_sha256=digest((root/'manifest.safe.json').read_bytes()),
        checks=['artifact and original-source hashes','JSONL/Parquet equality','actual tokenizer lengths and special tokens',
                'single-source block provenance and page bounds','group separation and unique released blocks',
                '1672-case exact/13-word fingerprint isolation','cross-split block 5-word Jaccard >=0.85 (>=40 words)'],
        splits=summaries,source_counts=source_counts,cross_split_near_duplicate_pairs=near_pairs,
        benchmark_source_families_excluded=sum(s['benchmark_source_family'] for s in catalog.values()),
        training_started=False,limitations=['Sampled visual QA, not a proof that every OCR formula is correct','No semantic benchmark decontamination guarantee'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path('CPT_resources/corpus_build_inputs_20260910'))
    parser.add_argument('--sources',type=Path,default=Path('CPT_resources/CPT-LOGISTS'))
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    report=verify(args.root,args.inputs,args.sources)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
