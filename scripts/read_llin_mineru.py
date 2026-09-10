"""Read only user-authorized MinerU UI exports; never fetch embedded image URLs."""
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup


def render_block(block):
    kind=block['type']; flags=[]
    if kind in ('image','chart'):
        return '',['image_or_chart_not_text']
    if kind in ('ref_text','header','footer','page_number'):
        flags.append('references_or_page_furniture')
    if kind in ('table','code'):
        children=[render_block(b) for b in block.get('blocks',[])]
        return '\n\n'.join(t for t,_ in children if t),[f for _,fs in children for f in fs]+(['code_layout_requires_review'] if kind=='code' else [])
    parts=[]
    for line in block.get('lines',[]):
        spans=[]
        for span in line.get('spans',[]):
            t=span.get('content','');typ=span.get('type','text')
            if typ in ('inline_equation','interline_equation'):
                if t.count('{')!=t.count('}') or not t.strip():flags.append('unbalanced_or_empty_formula')
                t=('$$\n'+t+'\n$$') if typ=='interline_equation' else '$'+t+'$'
            elif typ=='table':
                t=span.get('html','')
                soup=BeautifulSoup(t,'lxml')
                table=soup.find('table')
                if table is None or not table.find_all('tr'):flags.append('invalid_table')
                for node in soup.select('script,style,img'):node.decompose()
                t=str(table) if table else ''
            elif typ not in ('text','code'):
                flags.append('unsupported_span_'+typ)
            spans.append(t)
        parts.append(''.join(spans))
    text='\n'.join(parts).strip()
    if block.get('lines_deleted'):flags.append('merged_elsewhere_by_mineru')
    if kind=='title':text='# '+text
    if '\ufffd' in text or re.search(r'\(cid:\d+\)',text):flags.append('encoding_damage')
    if re.search(r'<\|(?:im_start|im_end|endoftext)\|>',text):flags.append('model_control_token')
    return text,flags


def read_mineru(source_name,source_id,parts,folder):
    from prepare_llin_logists_corpus import sha,STARTS,norm
    selected=sorted([p for p in parts if p['source']==source_name],key=lambda p:p['first_page'])
    if not selected:return None
    result=[];section=[];chapter='book';expected_page=1
    start=next((v for k,v in STARTS.items() if source_name.startswith(k)),1)
    for part in selected:
        assert part['source_sha256']==source_id and part['first_page']==expected_page
        file=folder/(Path(part['file']).stem+'.json')
        parsed=json.loads(file.read_text(encoding='utf-8'))
        assert len(parsed['pdf_info'])==part['last_page']-part['first_page']+1
        for offset,page in enumerate(parsed['pdf_info']):
            assert page['page_idx']==offset
            number=part['first_page']+offset
            for index,block in enumerate(page['para_blocks']):
                text,flags=render_block(block)
                text=norm(text)
                if number<start:flags.append('front_matter')
                if block['type']=='title':
                    title=text.lstrip('# ').strip()
                    # Numbered list items and repeated paper sections are not book chapters.
                    if re.match(r'(?i)^chapter\s+\d+',title):
                        chapter=title
                    section=[chapter,title] if title!=chapter and chapter!='book' else [title]
                    if re.search(r'(?i)^(references|bibliography|index|contents)$',title):flags.append('reference_heading')
                if len(text.strip())<2 and not flags:flags.append('empty_fragment')
                # Keep visual references with their figure in the audit archive, not plain-text release.
                if re.search(r'(?i)\b(?:figure|fig\.)\s*\d|as (?:shown|illustrated) in',text):flags.append('visual_dependency')
                if re.search(r'(?:\.\s*){5,}',text):flags.append('table_of_contents')
                if block['type']=='text' and re.fullmatch(r'[\divxIVX ]+',text):flags.append('page_furniture')
                if block.get('merge_prev') and result and not flags and not result[-1]['flags'] and result[-1].get('kind')=='text' and block['type']=='text':
                    prev=result[-1];prev['text']+=' '+text;prev['text_sha256']=sha(prev['text']);prev['block_id']=sha(prev['block_id']+'|'+str(number)+'|'+text)
                    prev.setdefault('additional_pages',[]).append(number);continue
                result.append(dict(block_id=sha(f'{source_id}:mineru:{number}:{index}:{text}'),text=text,text_sha256=sha(text),
                    page=number,bbox=block.get('bbox',[]),section=section.copy(),group_id=sha(source_id+'|'+chapter),
                    flags=sorted(set(flags)),kind=block['type'],extractor='MinerU '+str(parsed.get('_version_name'))))
        expected_page=part['last_page']+1
    return result,expected_page-1
