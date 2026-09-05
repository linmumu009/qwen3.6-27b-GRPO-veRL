"""Freeze the eight authorized, agent-reviewed examples for a mask-only diagnostic."""
import argparse
import hashlib
import json
import re
from pathlib import Path

REVIEWED = (
    '05d70a13e26c6c34aa026d82104680188481efffcd926494a813637e549f0fae',
    '02a9348a4f202e5e01e7f2f065172bcee3501da1c809c72e31a3b77b699ba55d',
    '003b502e5edadbc3dda404fa525a2eac01bf948b9d84f84a3acb8c4b5241c7cf',
    '01512c3a8ddccce75d4f20ace58cdc9c1cd864087365abd61644865152258ed4',
    '005d51199e20bdc47929af43b6d05e25dfdf0b7ab39e0b3498ed8d725d857de8',
    '0095fcec396e2a6d00b4e1d80712c536290594ea491b5b635f0f44be323e0421',
    '00d06b4a5050e38935bd7c40b504b677423707a7a36a3ba1b22d7eb6b45fccec',
    '012af34f30fe8e95448e3954ef8b89a41d5aef1055120dee20c11148dc1891f7',
)


def clean_label(text):
    return re.sub(r'^\s*(?:\(\d+\)\s*|-\s*(?=[A-Za-z]))', '', text).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    import pandas as pd
    from scripts.prepare_logistics_exam_cpt import read_jsonl, render_training_document, write_jsonl_private, write_json
    if args.output_dir.exists():
        raise ValueError('refusing overwrite')
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99':
        raise ValueError('source changed since review')
    if hashlib.sha256(args.candidates.read_bytes()).hexdigest() != '6b0913bb9e27283e82f89aec7bf1c40e2e4d65423cfa779d1dacc1bfad7632ac':
        raise ValueError('candidates changed since review')
    source = {r['item_hash']:r for r in read_jsonl(args.source)}
    candidates = {r['item_hash']:r for r in read_jsonl(args.candidates)}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows, facts = [], []
    for key in REVIEWED:
        row = dict(source[key])
        row['options'] = [clean_label(x) for x in row['options']]
        text = render_training_document(row,row['question'])
        rows.append({'item_hash':key,'arm':'reviewed8','text':text,'training_document':text,
                     'token_count':len(tokenizer.encode(text,add_special_tokens=False))})
        fact = candidates[key]['candidate']['text']
        if key == REVIEWED[-1]:
            fact = re.sub(r' in the [^.]+ chapter(?=\.)', '', fact)
            if 'chapter' in fact.lower():
                raise ValueError('chapter reference repair incomplete')
        facts.append({'item_hash':key,'text':fact,'agent_reviewed_for_gold_entailment':True,
                      'independently_fact_checked':False})
    write_jsonl_private(args.output_dir/'items.jsonl',rows)
    write_jsonl_private(args.output_dir/'reviewed-facts.jsonl',facts)
    path = args.output_dir/'same-text.parquet'
    pd.DataFrame(rows).to_parquet(path,index=False)
    import os
    os.chmod(path,0o600)
    write_json(args.output_dir/'manifest.safe.json',{'private_content_included':False,
        'source_hash':source_hash,'reviewed_item_hashes':list(REVIEWED),
        'items':8,'false_judgment_items_reviewed':0,'chapter_reference_repairs':1,
        'agent_reviewed':8,'human_reviewed':0,'independently_fact_checked':0,
        'training_text':'identical standalone historical QA with leading answer-list markers removed',
        'facts_are_separate_not_used_in_mask_ablation':True,
        'content_tokens':sum(r['token_count'] for r in rows),
        'parquet_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
