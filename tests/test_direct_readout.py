"""Model-free split, target and metric checks; numerical solves run on CUDA."""
import subprocess
import sys
from types import SimpleNamespace
import unittest
from qwen_one_two.training.direct_readout import fold, selected, targets, metrics, fit_operation, paired_values


def release(rows,purpose='training'):
    return SimpleNamespace(purpose=purpose,record=lambda k:rows[k])


def row(op,target,*,family='a',status='Assessed'):
    payload={'p_true':target} if op=='Noul' else {'score_interval':target}
    return {'operation':op,'family':family,'supervision':{'result':{'operation':op,'status':status,'payload':payload}}}


class DirectReadoutTests(unittest.TestCase):
    def test_family_assignment_and_scalar_eligibility(self):
        self.assertEqual([fold('same')]*3,[fold('same') for _ in range(3)])
        self.assertEqual({fold(str(i)) for i in range(100)},set(range(3)))
        r=release({'a':row('Noul',1),'b':row('Noul',None,status='Insufficient'),'c':row('Score',[.2,.2])})
        self.assertEqual(selected(r,['a','b','c'],'Noul'),['a'])
        r.purpose='locked'
        with self.assertRaises(ValueError):selected(r,['a'],'Noul')

    def test_fit_rejects_development_before_model_import(self):
        for purpose in ('development','locked'):
            with self.assertRaisesRegex(ValueError,'never fit'):
                fit_operation(release({},purpose),[],None)

    def test_no_interval_midpoint_substitution_and_no_tolerance_widening(self):
        r=release({'a':row('Score',[.5,.5]),'b':row('Score',[.4,.6])})
        with self.assertRaisesRegex(ValueError,'interval'):targets(r,['b'])
        self.assertEqual(metrics(r,['a'],[.5])['exact_hits'],1)
        self.assertEqual(metrics(r,['a'],[.500001])['exact_hits'],0)

    def test_noul_ties_classes_and_raw_error_are_visible(self):
        r=release({'a':row('Noul',1),'b':row('Noul',0),'c':row('Noul',0)})
        m=metrics(r,['a','b','c'],[.5,-1.,2.])
        self.assertEqual((m['correct'],m['true_positive'],m['true_negative'],m['ties']),(1,0,1,1))
        self.assertEqual(m['clipped_predictions'],2)
        self.assertGreater(m['raw_mse'],m['clipped_mse'])

    def test_pair_direction_does_not_replace_exact_correctness(self):
        r=release({'a':row('Score',[.2,.2]),'b':row('Score',[.8,.8])},'development')
        pair={'id':'p','ids':['a','b'],'operation':'Score','category':7}
        m=paired_values(r,[pair],{'a':.3,'b':.7})
        self.assertEqual(m['direction_correct'],1);self.assertEqual(m['both_scalar_correct'],0)
        self.assertEqual(paired_values(r,[pair],{'a':.3})['two_assessed_scalar_pairs'],0)

    def test_import_has_no_model_runtime(self):
        code="""
import sys
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','transformers','safetensors','numpy'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import scripts.diagnose_direct_readout
"""
        subprocess.run([sys.executable,'-c',code],check=True)


if __name__=='__main__':unittest.main()
