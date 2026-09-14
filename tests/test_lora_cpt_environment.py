import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('lora_pipeline', Path(__file__).parents[1]/'scripts/run_lora_cpt_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'fcntl':types.ModuleType('fcntl')}):
    spec.loader.exec_module(pipeline)


class EnvironmentTests(unittest.TestCase):
    def test_training_clears_all_inference_visibility_without_mutating_parent(self):
        original = dict(ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                        ASCEND_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0', PYTHONPATH='keep')
        result = pipeline.stage_environment(original, 'formal_training')
        self.assertEqual(result, {'PYTHONPATH':'keep'})
        self.assertEqual(original['ASCEND_VISIBLE_DEVICES'], '0')

    def test_evaluation_to_training_transition_does_not_restrict_16_rank_training(self):
        fake = types.ModuleType('run_logistics_cpt_curve_8x')
        fake.evaluation_env = lambda env: dict(env, ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        with patch.dict(sys.modules, {'run_logistics_cpt_curve_8x':fake}):
            evaluation = pipeline.stage_environment({'PYTHONPATH':'keep'}, 'gate_inference')
            self.assertIn('ASCEND_RT_VISIBLE_DEVICES', evaluation)
            training = pipeline.stage_environment(evaluation, 'formal_training')
            self.assertNotIn('ASCEND_RT_VISIBLE_DEVICES', training)
            self.assertIn('ASCEND_RT_VISIBLE_DEVICES', evaluation)


if __name__ == '__main__':
    unittest.main()
