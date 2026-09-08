import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_book_capability_batch import valid,indexed,good,CHECKS

class CapabilityTests(unittest.TestCase):
    def q(self):return dict(id='Q1',kind='condition',mode='closed',concept='test',question='What condition matters?',answer='A necessary condition.',rubric=['Necessary condition'],source_ids=['S001'])
    def test_valid(self):self.assertTrue(valid(self.q(),{'S001':'text'}))
    def test_context(self):
        q=self.q();q['question']='According to the source, what matters?'
        self.assertFalse(valid(q,{'S001':'text'}));q['mode']='evidence';self.assertTrue(valid(q,{'S001':'text'}))
    def test_citation_and_limit(self):
        q=self.q();q['source_ids']=[1];self.assertFalse(valid(q,{'S001':'text'}))
        q=self.q();q['answer']='word '*121;self.assertFalse(valid(q,{'S001':'text'}))
    def test_index_contract(self):
        self.assertIsNone(indexed({'items':[{'id':'Q1'},{'id':'Q1'}]},['Q1','Q2']))
        self.assertIsNone(indexed({'items':[{'id':'Q2'}]},['Q1']))
        self.assertEqual(set(indexed({'items':[{'id':'Q1'}]},['Q1'])),{'Q1'})
    def test_review(self):
        q=dict.fromkeys(CHECKS,True);q['issues']=[];self.assertTrue(good(q,CHECKS))
        q['supported']='true';self.assertFalse(good(q,CHECKS))

if __name__=='__main__':unittest.main()
