import zipfile
import json
from pathlib import Path
import pytest

from scripts.prepare_llin_logists_corpus import fingerprint_hit,join_lines,norm,read_zip,sha,NearDuplicates
from scripts.read_llin_mineru import render_block,read_mineru


def test_preserve_numeric_signs_units_and_negation():
    assert norm('Do not ship 10 m² below −5 °C.ﬁ')=='Do not ship 10 m² below −5 °C.fi'


def test_dehyphenation_requires_independent_word_evidence():
    assert join_lines(['manag-','ing inventory'],{'managing'})=='managing inventory'
    assert join_lines(['order-','picking'],{'picking'})=='order-picking'


def test_benchmark_hash_window():
    words=' '.join(f'w{i}' for i in range(13))
    assert fingerprint_hit('prefix '+words+' suffix',{'ngrams':{sha(words)},'exact':set()})


def test_near_duplicate_keeps_first_and_distinct_passage():
    screen=NearDuplicates();original=' '.join(f'w{i}' for i in range(100))
    assert not screen.add(original)
    assert screen.add(original.replace('w50 ','changed '))
    assert not screen.add(' '.join(f'other{i}' for i in range(100)))


def test_zip_selects_subsections_without_executing_scripts(tmp_path):
    p=tmp_path/'book.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('book/s05-01-flow.html','<div class="section"><script>DO NOT TRAIN THIS</script><p>Material flows from receiving to putaway.</p></div>')
        z.writestr('book/s05-flow.html','<p>Duplicated full chapter</p>')
    blocks,_=read_zip(p,'source')
    assert [b['text'] for b in blocks]==['Material flows from receiving to putaway.']


def test_mineru_keeps_math_and_rejects_unbalanced_formula():
    block={'type':'interline_equation','lines':[{'spans':[{'type':'interline_equation','content':r'x_{t+1}=x_t-2'}]}]}
    text,flags=render_block(block)
    assert text=='$$\nx_{t+1}=x_t-2\n$$' and not flags
    block['lines'][0]['spans'][0]['content']='x_{t'
    assert 'unbalanced_or_empty_formula' in render_block(block)[1]


def test_mineru_preserves_table_cells_and_merged_header():
    table='<table><tr><td colspan="2">Quantity</td></tr><tr><td>A</td><td>12</td></tr></table>'
    text,flags=render_block({'type':'table_body','lines':[{'spans':[{'type':'table','html':table}]}]})
    assert 'colspan="2"' in text and '<td>12</td>' in text and not flags


def test_mineru_contiguous_parts_merge_paragraph_and_track_pages(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    parts=[]
    for page,text in [(1,'The shipment'),(2,'arrives tomorrow.')]:
        block={'type':'text','merge_prev':page==2,'lines':[{'spans':[{'content':text}]}]}
        data={'pdf_info':[{'page_idx':0,'para_blocks':[block]}],'_version_name':'test'}
        (tmp_path/f'part{page}.json').write_text(json.dumps(data),encoding='utf-8')
        parts.append(dict(source='book.pdf',source_sha256='source',file=f'part{page}.pdf',first_page=page,last_page=page))
    blocks,pages=read_mineru('book.pdf','source',parts,tmp_path)
    assert pages==2 and len(blocks)==1
    assert blocks[0]['text']=='The shipment arrives tomorrow.'
    assert blocks[0]['additional_pages']==[2]
    parts[1]['first_page']=3
    with pytest.raises(AssertionError):read_mineru('book.pdf','source',parts,tmp_path)


def test_code_layout_is_archived_with_content_not_silently_dropped():
    block={'type':'code','blocks':[{'type':'code_body','lines':[{'spans':[{'content':'If demand > stock then stop'}]}]}]}
    text,flags=render_block(block)
    assert 'demand > stock' in text and 'code_layout_requires_review' in flags
