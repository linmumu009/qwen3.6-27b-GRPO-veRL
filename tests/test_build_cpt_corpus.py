from types import SimpleNamespace

from scripts.build_cpt_corpus import html_blocks, make_chunks, normalize, overlap, rights, sha


def test_normalization_preserves_numeric_and_negation_semantics():
    assert normalize('  Do not ship 10 m² at −5 °C.\u00a0')=='Do not ship 10 m² at −5 °C.'


def test_rights_do_not_follow_spoofed_source_label():
    assert rights(dict(id='erp_operations',url='https://example.com/erpnext/stock'))=='rights_review_required'
    assert rights(dict(id='erp_operations',url='https://docs.frappe.io/erpnext/stock'))=='release_cc_by_sa'


def test_article_selector_excludes_navigation_and_preserves_conditions():
    raw=b'<title>Receiving</title><nav>fake inventory rule</nav><article><header>Copy page</header><div id="wiki-content"><h2>Validate</h2><p>Do not post until inspection is complete.</p><ul><li>Sales Order</li></ul></div></article>'
    title,blocks,flags=html_blocks(raw,dict(id='erp_operations',title='x'))
    assert title=='Receiving' and not flags
    assert [b['text'] for b in blocks]==['Do not post until inspection is complete.','- Sales Order']


def test_code_indentation_and_table_rows_survive():
    raw=b'<title>X</title><div id="wiki-content"><pre>if ok:\n    ship()</pre><table><tr><td>A</td><td>10</td></tr><tr><td>B</td><td>20</td></tr></table></div>'
    _,blocks,_=html_blocks(raw,dict(id='erp_operations',title='x'))
    assert '    ship()' in blocks[0]['text']
    assert 'A\t10\nB\t20' in blocks[1]['text']


def test_merged_table_is_flagged_instead_of_flattened_as_fact():
    _,blocks,_=html_blocks(b'<div id="wiki-content"><table><tr><td colspan="2">Total</td></tr></table></div>',dict(id='erp_operations',title='x'))
    assert blocks[0]['flags']==['complex_table_requires_review']


def test_skipped_heading_levels_do_not_nest_siblings():
    raw=b'<div id="wiki-content"><h2>Basics</h2><h4>Company</h4><p>First.</p><h4>Customer</h4><p>Second.</p><h2>Related Topics</h2><p>Other links</p></div>'
    _,blocks,_=html_blocks(raw,dict(id='erp_operations',title='x'))
    assert blocks[1]['section']==['Basics','Customer']
    assert blocks[2]['flags']==['related_topics_navigation']


def test_contamination_detects_window_inside_longer_paragraph():
    text=' '.join('word'+str(i) for i in range(13))
    assert overlap('prefix '+text+' suffix',dict(ngrams={sha(text)},exact=set()))==1


def test_chunking_keeps_source_section_and_limit():
    tok=SimpleNamespace(encode=lambda text,**kwargs:SimpleNamespace(ids=text.split()))
    blocks=[dict(text='one two three four',block_id='a',section=['A']),dict(text='five six seven eight',block_id='b',section=['B'])]
    chunks=make_chunks(blocks,'Test',tok,limit=20)
    assert len(chunks)==1
    assert chunks[0]['block_ids']==['a','b']
    assert '\nA\n' in chunks[0]['text'] and '\nB\n' in chunks[0]['text']
    assert all(c['content_tokens']<=20 for c in chunks)
