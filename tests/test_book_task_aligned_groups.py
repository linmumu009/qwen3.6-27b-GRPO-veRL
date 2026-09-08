import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_book_task_aligned_groups import KINDS, FIELDS, choose_tasks, valid_group, solver_input, solver_valid, audit_valid, audit_solution, HELDOUT


def fixture():
    qs=[dict(kind=k,question='Question '+k,answer='Supported short conclusion.',rubric=['Essential point'],source_ids=['S001']) for k in KINDS]
    qs[-1].update(options=['Candidate '+str(i) for i in range(6)],correct_indices=[1,3],option_reasons=['Supported distinction']*6)
    return dict(concept='Synthetic rule',rule='An explicit rule',source_ids=['S001'],questions=qs)


class GroupTests(unittest.TestCase):
    def test_split_and_capacity(self):
        records=[dict(record_id=f'{c}-{i}',chapter=c,text='x'*2000) for c in range(1,40) for i in range(4)]
        tasks=choose_tasks(records)
        self.assertEqual(len(tasks),120)
        self.assertEqual(sum(t['split']=='dev' for t in tasks),30)
        self.assertTrue(all((t['source']['chapter'] in HELDOUT)==(t['split']=='dev') for t in tasks))
        self.assertEqual(tasks,choose_tasks(records))
        with self.assertRaises(ValueError):choose_tasks(records[:1])

    def test_four_tasks_and_strict_indices(self):
        g=fixture();self.assertTrue(valid_group(g,{'S001':'text'}))
        for value in ([True],[1,1],[6],[]):
            x=copy.deepcopy(g);x['questions'][-1]['correct_indices']=value
            self.assertFalse(valid_group(x,{'S001':'text'}))
        g['questions'][-1]['options'][0]='All of the above'
        self.assertFalse(valid_group(g,{'S001':'text'}))

    def test_solver_blinding_and_reference(self):
        g=fixture();payload,mapping=solver_input(g,{'S001':'text'},'test')
        self.assertEqual(sorted(mapping),list(range(6)))
        for q in payload['questions']:
            self.assertNotIn('answer',q);self.assertNotIn('correct_indices',q);self.assertNotIn('rubric',q)
        self.assertEqual(payload['questions'][-1]['reference'],{'S001':'text'})

    def test_audit_strict_booleans(self):
        v=dict(same_concept=True,questions=[dict(kind=k,**dict.fromkeys(FIELDS+('solver_equivalent',),True),source_ids=['S001'],issues=[]) for k in KINDS])
        self.assertTrue(audit_valid(v,{'S001':'text'}))
        v['questions'][0]['supported']=1
        self.assertFalse(audit_valid(v,{'S001':'text'}))

    def test_embedded_options_rejected(self):
        g=fixture();g['questions'][-1]['question']+=' Candidate 0'
        self.assertFalse(valid_group(g,{'S001':'text'}))

    def test_closed_question_cannot_depend_on_missing_source(self):
        g=fixture();g['questions'][0]['question']='According to the source, define the rule.'
        self.assertFalse(valid_group(g,{'S001':'text'}))

    def test_auditor_receives_remapped_text_not_ambiguous_indices(self):
        value={'questions':[{}, {}, {}, dict(selected_indices=[0,2],source_ids=['S001'],answer='Index prose')]}
        r=audit_solution(value,fixture(),[3,0,1,2,4,5])
        self.assertEqual(r['questions'][-1]['selected_options'],['Candidate 3','Candidate 1'])
        self.assertNotIn('answer',r['questions'][-1]);self.assertNotIn('selected_indices',r['questions'][-1])
        self.assertIn('answer',value['questions'][-1])

    def test_solver_unresolved_rejected(self):
        v={'questions':[dict(kind=k,answer='x',source_ids=['S001'],answerable=True,unambiguous=True,selected_indices=[2] if k=='evidence_selection' else []) for k in KINDS]}
        self.assertTrue(solver_valid(v,{'S001':'text'},list(range(6))))
        v['questions'][-1]['answerable']=False
        self.assertFalse(solver_valid(v,{'S001':'text'},list(range(6))))

    def test_selection_does_not_require_redundant_answer_prose(self):
        v={'questions':[dict(kind=k,answer='x',source_ids=['S001'],answerable=True,unambiguous=True,selected_indices=[2] if k=='evidence_selection' else []) for k in KINDS]}
        del v['questions'][-1]['answer']
        self.assertTrue(solver_valid(v,{'S001':'text'},list(range(6))))
        del v['questions'][0]['answer']
        self.assertFalse(solver_valid(v,{'S001':'text'},list(range(6))))

if __name__=='__main__':unittest.main()
