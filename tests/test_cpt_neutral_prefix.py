import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_cpt_neutral_prefix import PREFIX,MARKER,neutral_question,neutral_messages
from summarize_cpt_neutral_prefix import aggregate


def test_preserves_every_suffix_character_and_option():
    suffix='  Definition with a condition, NOT an exception.\nTrailing space '
    original=PREFIX+MARKER+suffix
    assert neutral_question(original)==MARKER+suffix
    options=[f'choice {i}' for i in range(269)]
    result=neutral_messages(original,options)
    assert result[1]['content']=='Question:\n'+MARKER+suffix+'\n\nOptions:\n'+'\n'.join(f'[{i}] {v}' for i,v in enumerate(options))


@pytest.mark.parametrize('question',[MARKER+' x',PREFIX+MARKER,PREFIX+MARKER+' a '+MARKER+' b','Other domain '+MARKER+' x'])
def test_unknown_or_ambiguous_boundary_fails(question):
    with pytest.raises(ValueError):neutral_question(question)


def test_prompt_api_rejects_labels():
    with pytest.raises(TypeError):neutral_messages(PREFIX+MARKER+' definition',['a'],expected=[0])


def test_net_effect_includes_retention_losses():
    def result(correct):return dict(correct=correct,stable_correct=correct,unstable=False,invalid=0,truncated=0)
    rows=[dict(id='a',source_status='A',old_target=True,before=result(False),after=result(True)),
          dict(id='b',source_status='not_independently_reviewed',old_target=False,before=result(True),after=result(False))]
    table=aggregate(rows)
    assert (table[0]['gains'],table[0]['losses'],table[0]['net'])==(1,1,0)
    assert table[3]['n']==1 and table[3]['gains']==1
