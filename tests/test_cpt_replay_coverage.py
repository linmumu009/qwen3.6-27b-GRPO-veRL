from copy import deepcopy
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_cpt_replay_coverage import cases_from_reviewed

def records():
    return [dict(id=str(i),split='train',task=dict(form='definition_discrimination',
        question=f'question {i}',options=['a','b'],correct_indices=[i%2])) for i in range(135)]

def test_replay_probe_excludes_dev_and_preserves_all_original_fields():
    rows=records();dev=deepcopy(rows[0]);dev.update(id='heldout',split='dev')
    cases=cases_from_reviewed(rows+[dev])
    assert len(cases)==135 and all(c['source_id']!='p1-replay-heldout' for c in cases)
    for c,r in zip(cases,rows):
        assert c['question']==r['task']['question'] and c['options']==r['task']['options']
        assert c['expected']==r['task']['correct_indices'] and c['dataset']=='replay_training_fit'

def test_incomplete_or_duplicate_replay_cannot_be_published():
    rows=records()
    with pytest.raises(AssertionError):cases_from_reviewed(rows[:-1])
    rows[-1]=rows[0]
    with pytest.raises(AssertionError):cases_from_reviewed(rows)
