import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cpt_selection_diagnostic import messages,strict_supported,score
from cpt_two_family import digest


def fixture():
    groups=[]
    requests=[]
    for task in ("MT1","MT2","MT3","MT4"):
        conditions=[]
        for shift in range(4):
            order=[(i+shift)%4 for i in range(4)]
            conditions.append(dict(order=order,expected=[i for i,j in enumerate(order) if j in (0,1)]))
        groups.append(dict(id=task,conditions=conditions))
    for repeat in range(3):
        for g in groups:
            for shift in (range(4) if repeat==0 else (0,)):
                for form,pos in [("S",None),("V",None)]+[("B",i) for i in range(4)]:
                    requests.append(dict(id=f'{g["id"]}:{shift}:{repeat}:{form}:{pos}',task=g['id'],shift=shift,repeat=repeat,form=form,position=pos,messages=[dict(role="user",content="fixed")]))
    packet=dict(groups=groups,requests=requests)
    rows=[]
    for r in requests:
        g=next(g for g in groups if g['id']==r['task'])
        expected=g['conditions'][r['shift']]['expected']
        values=[i in expected for i in range(4)]
        value={'answers':expected} if r['form']=='S' else {'supported':values if r['form']=='V' else values[r['position']]}
        rows.append(dict(id=r['id'],text=json.dumps(value),output_tokens=10,finish_reason='stop',messages_sha256=digest(r['messages'])))
    return packet,rows


def test_strict_boolean_schema():
    assert strict_supported('{"supported":false}','B') is False
    assert strict_supported('{"supported":[true,false,true,false]}','V')==[True,False,True,False]
    for text in ['{"supported":0}','{"supported":"false"}','{"supported":true,"supported":false}','{"supported":true,"other":1}','{"supported":NaN}']:
        assert strict_supported(text,'B') is None
    for text in ['{"supported":[true]}','{"supported":[1,0,1,0]}','{"supported":[true,false,true,null]}']:
        assert strict_supported(text,'V') is None


def test_all_calls_and_binary_reconstruction():
    packet,rows=fixture()
    ledger,summary=score(packet,rows)
    assert len(ledger)==72
    assert [summary[f]['calls'] for f in ('S','V','B')]==[24,24,96]
    assert all(summary[f]['stable_tasks']==4 for f in summary)
    row=next(r for r in rows if r['id']=='MT1:0:0:B:0')
    row['text']='{"supported":false}'
    _,summary=score(packet,rows)
    assert summary['B']['main_complete_sets_correct']==15
    assert summary['B']['stable_tasks']==3 and summary['B']['observed_omissions']==1
    assert summary['S']['stable_tasks']==4


def test_truncated_binary_is_not_a_complete_set():
    packet,rows=fixture()
    rows[2]['finish_reason']='length'
    _,summary=score(packet,rows)
    assert summary['B']['truncated']==1 and summary['B']['main_complete_sets_correct']==15
    assert summary['B']['tasks'][0]['first_three_valid'] is False


def test_prompt_change_preserves_options_and_original():
    original=[dict(role='system',content='old'),dict(role='user',content='Context. Select all correct statements.\nOptions:\n[0] first\n[1] second\n[2] third\n[3] fourth')]
    before=copy.deepcopy(original)
    assert messages(original,'S')==original
    for form,position in [('V',None),('B',2)]:
        out=messages(original,form,position)
        assert '\nOptions:\n[0] first\n[1] second\n[2] third\n[3] fourth' in out[1]['content']
        assert 'Assess the statements under the stated scope.' in out[1]['content']
    assert original==before
