"""Model-free safeguards for receipt accounting; never execute a model."""
import subprocess
import sys
from types import SimpleNamespace
import unittest

from scripts.audit_latent_semantics import audit, recipe, scalar_summary, unique


def row(op, payload):
    return {'id': 'case', 'family': 'family', 'operation': op,
            'construction': {'template_version': 'controlled-data-v1'},
            'semantics': {}, 'supervision': {'result': {'operation': op, 'status': 'Assessed', 'payload': payload}}}


class LatentSemanticAuditTests(unittest.TestCase):
    def test_no_model_library_imports(self):
        program = '''
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers'}:
   raise RuntimeError('model library forbidden: '+name)
sys.meta_path.insert(0,Block())
import scripts.audit_latent_semantics
'''
        subprocess.run([sys.executable,'-c',program],check=True)

    def test_locked_rejected_before_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'no locked'):
            audit(SimpleNamespace(purpose='training'),SimpleNamespace(purpose='locked'))

    def test_duplicate_receipts_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            unique([{'id':'one'},{'id':'one'}],'id')

    def test_scalar_threshold_point_and_one_class_scope(self):
        r = row('Noul',{'p_true':1})
        result = scalar_summary([r],{'case':.5})
        self.assertEqual(result['correct'],0)
        self.assertIsNone(result['auc']); self.assertIsNone(result['balanced_accuracy'])
        r = row('Score',{'score_interval':[.2,.3]})
        with self.assertRaisesRegex(ValueError, 'point-only'):
            scalar_summary([r],{'case':.25})
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            scalar_summary([r],{'case':float('nan')})

    def test_strict_count_units_and_reviewed_formula(self):
        r = row('Score',{'score_interval':[.25,.25]})
        r['model_inputs'] = {'evidence': {'facts':['The requirement is 200 panels.','50 panels are complete.']}}
        r['semantics'] = {'question':{},'rubric':{'conditions':'Completed panels divided by required panels','direction':'higher'}}
        self.assertEqual(recipe(r)['denominator'],200)
        r['model_inputs']['evidence']['facts'][1] = '50 seats are complete.'
        with self.assertRaisesRegex(ValueError, 'count/unit'):
            recipe(r)
        r['model_inputs']['evidence']['facts'][1] = '51 panels are complete.'
        with self.assertRaisesRegex(ValueError, 'formula mismatch'):
            recipe(r)


if __name__ == '__main__':
    unittest.main()
