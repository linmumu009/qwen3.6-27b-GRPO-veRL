import importlib.util
from pathlib import Path
import unittest


spec=importlib.util.spec_from_file_location('overlap',Path(__file__).parents[1]/'scripts/audit_cpt_logistika_overlap.py')
overlap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)


class OverlapTests(unittest.TestCase):
    def test_numbers_do_not_hide_same_question(self):
        self.assertEqual(overlap.norm('Move 120.5 tons via 2 ports.'),overlap.norm('Move 80 tons via 5 ports.'))

    def test_options_and_user_messages_are_retained_without_assistant_answers(self):
        corpus=[]
        overlap.collect([{'question':'Which rule?', 'options':{'A':'Actual condition','B':'Competing condition'}},
                         {'messages':[{'role':'user','content':'Original training task'},
                                      {'role':'assistant','content':'PRIVATE_ANSWER'}]}], 'synthetic', corpus)
        self.assertEqual(len(corpus),2)
        self.assertEqual(corpus[0]['options'],['Actual condition','Competing condition'])
        self.assertNotIn('PRIVATE_ANSWER',str(corpus))


if __name__=='__main__':unittest.main()
