"""Model-free input coverage, provenance and accounting regression checks."""
from copy import deepcopy
import subprocess
import sys
import unittest
from types import SimpleNamespace
from qwen_one_two.training.simple_latent import bins, assert_input_match, paired_outcomes


class SimpleLatentTests(unittest.TestCase):
    def test_bins_cover_tokens_once_for_every_supported_shape(self):
        for count in (2,4,8):
            for length in range(count,257):
                spans=bins(length,count)
                self.assertEqual(len(spans),count)
                self.assertEqual([i for a,b in spans for i in range(a,b)],list(range(length)))
                self.assertTrue(all(a<b for a,b in spans))

    def test_bins_reject_empty_truncated_or_invalid_lengths(self):
        for args in ((0,2),(3,4),(257,8),(True,1),(10,0)):
            with self.assertRaises(ValueError):bins(*args)

    def test_helpers_and_runner_do_not_import_model_libraries(self):
        code="""
import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','safetensors','numpy'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import scripts.compare_latent_inputs
"""
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_accounting_and_evidence_changes_rejected(self):
        case={'id':'x','record_sha256':'r','view_sha256':'v','runtime':{
            'prefills':1,'lineage':1,'held_positions':0,'total':12,'P':5,'B':4,'A':0,'H':3,
            'predicted_budget':0,'granted_budget':512,'override':512,
            'events':[{'label':'request','count':4}]},
            'construction_audit':[{'local_source':0,'tensor_hash':'original','offsets':[[0,2]]}]}
        assert_input_match(deepcopy(case),case)
        for name, value in (('total',13),('prefills',2),('granted_budget',128)):
            changed=deepcopy(case);changed['runtime'][name]=value
            with self.assertRaises(ValueError):assert_input_match(changed,case)
        changed=deepcopy(case);changed['construction_audit'][0]['tensor_hash']='different'
        with self.assertRaisesRegex(ValueError,'source evidence'):assert_input_match(changed,case)

    def test_incomplete_paired_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'coverage'):
            paired_outcomes(None,[{'id':'a'}],[{'id':'b'}])

    def test_paired_wins_losses_and_failures_keep_denominators(self):
        row={'operation':'Noul','supervision':{'result':{'operation':'Noul','status':'Assessed','payload':{'p_true':1}}}}
        flags={k:True for k in ('latent_request','latent_return','bound_snapshot','candidate_binding',
                               'consumed','single_prefill','cache_continuity','accounting')}
        def case(key,passed):
            return {'id':key,'actual':{'status':'Assessed','value':.9,'selected':None,'reason_present':False},
                    'transport':flags,'failure':None if passed else 'invalid result'}
        controls=[case(str(i),ok) for i,ok in enumerate((True,True,False,False))]
        candidates=[case(str(i),ok) for i,ok in enumerate((True,False,True,False))]
        result=paired_outcomes(SimpleNamespace(record=lambda key:row),controls,candidates)['Noul']
        self.assertEqual(result,{'cases':4,'both_pass':1,'control_only':1,'candidate_only':1,'both_fail':1})


if __name__=='__main__':unittest.main()
