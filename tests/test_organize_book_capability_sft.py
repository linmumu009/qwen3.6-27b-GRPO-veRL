import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from organize_book_capability_sft import valid

class OrganizationTests(unittest.TestCase):
    def test_minimal_contract(self):
        q=dict(decision='keep',answer='A concise answer.',rubric=['Necessary point'],source_ids=['S001'])
        self.assertTrue(valid(q,{'S001':'source'}))
        for change in (dict(decision='reject'),dict(answer='word '*91),dict(rubric=[]),dict(source_ids=[1])):
            self.assertFalse(valid(dict(q,**change),{'S001':'source'}))

if __name__=='__main__':unittest.main()
