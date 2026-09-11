import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_logists_full_pipeline import audit_training
import prepare_llin_logists_corpus as prep


def test_full_epoch_budget_and_missing_tail_rejected():
    lengths=[393]*4706+[1849588-393*4706]
    metrics={i+1:{'train/global_tokens':sum(lengths[i*9:(i+1)*9]),'train/loss':2.,'train/grad_norm':1.,'train/lr':5e-7} for i in range(523)}
    assert audit_training(metrics,lengths)['steps']==523
    metrics.pop(523)
    with pytest.raises(AssertionError):audit_training(metrics,lengths)


def test_all_materials_mode_keeps_source_families_and_matching_text(tmp_path,monkeypatch):
    inp=tmp_path/'input';inp.mkdir()
    names=['43_Book_test.pdf','MITx_test.pdf','KS-GQ-test.pdf','challenges-and-test.pdf']+[f'book{i}.pdf' for i in range(9)]
    for name in names:(inp/name).write_bytes(name.encode())
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_text('{}')
    monkeypatch.setattr(prep,'TOKENIZER_SHA',prep.sha(tokenizer.read_bytes()))
    fake=SimpleNamespace(no_padding=lambda:None,no_truncation=lambda:None,encode=lambda text,**kw:SimpleNamespace(ids=list(range(len(text.split())))))
    monkeypatch.setattr(prep,'Tokenizer',SimpleNamespace(from_file=lambda _:fake))
    common=' '.join(f'match{i}' for i in range(13))
    fp=tmp_path/'fingerprints.gz';fp.write_bytes(gzip.compress(json.dumps(dict(cases=1672,raw_text_included=False,ngrams=[prep.sha(common)],exact=[],source_sha256='test')).encode()))
    def extract(path,sid):
        text=common+' '+' '.join(f'{sid}word{i}' for i in range(60))
        return [dict(block_id=prep.sha(text),text=text,text_sha256=prep.sha(text),page=1,group_id=sid,section=[],flags=[])],1
    monkeypatch.setattr(prep,'read_pdf',extract)
    args=SimpleNamespace(input=inp,out=tmp_path/'out',cache=tmp_path/'cache',tokenizer=tokenizer,fingerprints=fp,parts=None,mineru_dir=None,all_materials=True)
    prep.build(args)
    rows=[json.loads(line) for line in (args.out/'train.jsonl').read_text().splitlines()]
    assert {r['source_file'] for r in rows}==set(names)
    assert all(common in r['text'] for r in rows)
    manifest=json.loads((args.out/'manifest.safe.json').read_text())
    assert manifest['splits']['validation']['records']==0
    assert not any('benchmark' in reason for reason in manifest['quarantine_reasons'])
