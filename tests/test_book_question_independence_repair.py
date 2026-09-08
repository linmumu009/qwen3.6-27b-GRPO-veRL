import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from repair_book_question_independence import valid_candidate, checks, FIELDS


class RepairContractTests(unittest.TestCase):
    def candidate(self):
        return dict(mode='closed', reason='A complete technical question', question='What is inventory turnover?', answer='A turnover ratio.', rubric=['Define the ratio'], source_ids=['S001'])

    def test_valid_candidate(self):
        self.assertTrue(valid_candidate(self.candidate(), {'S001': 'test'}))

    def test_missing_context_closed_rejected(self):
        v=self.candidate(); v['question']='According to the source, what is inventory turnover?'
        self.assertFalse(valid_candidate(v, {'S001': 'test'}))
        v['mode']='evidence'
        self.assertTrue(valid_candidate(v, {'S001': 'test'}))

    def test_bad_citation_and_rubric(self):
        for ids in ([1], ['001'], [], ['S001','S001'], ['S002']):
            v=self.candidate(); v['source_ids']=ids
            self.assertFalse(valid_candidate(v, {'S001': 'test'}))
        v=self.candidate(); v['rubric']=['']
        self.assertFalse(valid_candidate(v, {'S001': 'test'}))

    def test_review_fails_closed(self):
        v=dict.fromkeys(FIELDS,True); v['issues']=[]
        self.assertTrue(checks(v,FIELDS))
        for field in FIELDS:
            bad=dict(v); bad[field]='true'
            self.assertFalse(checks(bad,FIELDS))
        v['issues']=['uncertain']
        self.assertFalse(checks(v,FIELDS))


if __name__=='__main__': unittest.main()
