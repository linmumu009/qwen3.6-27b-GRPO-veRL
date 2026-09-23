"""Failure recovery must not silently spend the frozen generation budget twice."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('remaining_runner',SCRIPTS/'run_cpt_remaining_diagnostic.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class JournalTests(unittest.TestCase):
    def test_unresolved_generation_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);old=base/'old';out=base/'out';package=base/'package'
            (old/'step120_direct').mkdir(parents=True);out.mkdir();package.mkdir()
            identity={'path':str(base/'model')}
            (old/'step120_direct/identity.safe.json').write_text(json.dumps(identity))
            packet={'groups':[{'id':'synthetic','source_text':'Synthetic evidence.'}],
                'tasks':[dict(id='synthetic__'+f,group='synthetic',form=f,question='Select all correct statements.',
                    options=['a','b','c','d'],correct_indices=[0,1]) for f in ['scope','competing_rules','application','counterexample']]}
            (package/'diagnostic.private.json').write_text(json.dumps(packet))
            (package/'execution.safe.json').write_text(json.dumps({'packet_sha256':runner.sha(package/'diagnostic.private.json')}))
            class Tokenizer:
                @staticmethod
                def from_pretrained(*args,**kwargs):return Tokenizer()
                def apply_chat_template(self,*args,**kwargs):return 'synthetic prompt'
                def encode(self,*args,**kwargs):return [1,2,3]
            class FailingLLM:
                calls=0
                def __init__(self,**kwargs):pass
                def generate(self,*args,**kwargs):
                    FailingLLM.calls+=1
                    raise RuntimeError('Simulated worker failure after reservation')
            modules={'transformers':types.SimpleNamespace(AutoTokenizer=Tokenizer),
                'vllm':types.SimpleNamespace(LLM=FailingLLM,SamplingParams=lambda **kw:kw),
                'run_cpt_formal_transfer':types.SimpleNamespace(model_identity=lambda p:identity)}
            with patch.dict(sys.modules,modules),patch.object(runner,'OLD',old),patch.object(runner,'OUT',out):
                with self.assertRaisesRegex(RuntimeError,'Simulated'):runner.worker(package,'step120_current')
                self.assertTrue((out/'step120_current/batch-000.reserved.safe.json').exists())
                with self.assertRaisesRegex(AssertionError,'Unresolved'):runner.worker(package,'step120_current')
                self.assertEqual(FailingLLM.calls,1)
                self.assertFalse((out/'step120_current/completed.safe.json').exists())


if __name__=='__main__':unittest.main()
