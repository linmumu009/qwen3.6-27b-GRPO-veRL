"""Independent release checks; emits aggregate evidence, never benchmark text."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify(root, tokenizer_dir, fingerprint_path):
    manifest=json.loads((root/'manifest.safe.json').read_text(encoding='utf-8'))
    for name,expected in manifest['artifacts'].items():
        assert digest((root/name).read_bytes())==expected, name
    assert digest((tokenizer_dir/'tokenizer.json').read_bytes())==manifest['tokenizer_sha256']
    assert digest(fingerprint_path.read_bytes())==manifest['fingerprint_file_sha256']
    fp=json.loads(gzip.decompress(fingerprint_path.read_bytes()))
    hashes=set(fp['ngrams']);exact=set(fp['exact'])
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    assert tokenizer.eos_token_id==248046
    docs={x['source_id']:x for x in read_rows(root/'staging_documents.jsonl')}
    families={};signatures={};counts={};all_ids=set();maximum=0;minimum=4096
    for split in ('train','validation'):
        rows=read_rows(root/(split+'.jsonl'))
        assert rows==pq.read_table(root/(split+'.parquet')).to_pylist()
        families[split]={r['family_id'] for r in rows};signatures[split]=[]
        total=0
        for row in rows:
            assert row['id'] not in all_ids
            all_ids.add(row['id'])
            assert digest(row['text'].encode())==row['text_sha256']
            doc=docs[row['source_id']]
            assert doc['rights']=='release_cc_by_sa' and not doc['flags']
            assert row['source_url']==doc['source_url'] and row['raw_sha256']==doc['raw_sha256']
            blocks={b['block_id']:b for b in doc['blocks']}
            positions=[];body=[]
            for bid in row['block_ids']:
                block=blocks[bid]
                assert not block['flags']
                assert digest(block['text'].encode())==block['text_sha256']
                start=row['text'].find(block['text'], positions[-1] if positions else 0)
                assert start>=0, bid
                positions.append(start+len(block['text']));body.append(block['text'])
            ids=tokenizer.encode(row['text'],add_special_tokens=False)
            assert len(ids)==row['content_tokens']
            assert len(ids)+1==row['sequence_tokens']==row['token_count']<=4096
            assert row['eos_token_id']==tokenizer.eos_token_id and row['eos_added_by_loader']
            assert not set(ids)&set(tokenizer.all_special_ids)
            maximum=max(maximum,len(ids)+1);minimum=min(minimum,len(ids)+1);total+=len(ids)
            ws=re.findall(r'[a-z0-9]+',row['text'].lower())
            assert not {digest(' '.join(ws[i:i+13]).encode()) for i in range(max(0,len(ws)-12))}&hashes
            assert digest(' '.join(ws).encode()) not in exact
            ws=re.findall(r'[a-z0-9]+',' '.join(body).lower())
            signatures[split].append({' '.join(ws[i:i+5]) for i in range(max(0,len(ws)-4))})
        assert len(rows)==manifest['splits'][split]['records']
        assert total==manifest['splits'][split]['content_tokens']
        counts[split]=dict(records=len(rows),content_tokens=total)
    assert not families['train']&families['validation']
    similarity=max(len(a&b)/max(1,len(a|b)) for a in signatures['train'] for b in signatures['validation'])
    assert similarity<.85
    return dict(status='PASS',scope='artifact integrity, provenance, tokenization, exact decontamination and split checks',
        manifest_sha256=digest((root/'manifest.safe.json').read_bytes()),splits=counts,
        min_sequence_tokens=minimum,max_sequence_tokens=maximum,eos_token_id=tokenizer.eos_token_id,
        cross_split_max_5gram_jaccard=similarity,parquet_roundtrip=True,all_blocks_traceable=True,
        benchmark_exact_matches=0,semantic_contamination_proven_absent=False,
        factual_expert_review_complete=False,training_started=False,
        dependencies={n:importlib.metadata.version(n) for n in ('transformers','tokenizers','pyarrow','beautifulsoup4','lxml')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--tokenizer-dir',type=Path,required=True)
    parser.add_argument('--fingerprints',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.corpus,args.tokenizer_dir,args.fingerprints)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
