"""Policy, exposure and routing tests; no CPU model execution."""
from collections import Counter
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from qwen_one_two.codec.schema import Port
from qwen_one_two.training.receiver_repair import training_ids, visit_ids, installed
from qwen_one_two.training.receiver_comparison import value_losses


class ReceiverRepairTests(unittest.TestCase):
    def test_value_only_removes_status_without_rescaling_or_mutating_values(self):
        status, noul, score = object(), object(), object()
        for source, expected in (({'status':status},{}), ({'status':status,'noul':noul},{'noul':noul}),
                                 ({'status':status,'score':score},{'score':score}), ({},{})):
            original = dict(source)
            self.assertEqual(value_losses(source),expected)
            self.assertEqual(source,original)
        with self.assertRaises(ValueError): value_losses({'choice':object()})

    def test_training_only_and_complete_exposure(self):
        data = {str(i): {'operation': 'Noul' if i % 2 else 'Score', 'training_category': 'receiver'} for i in range(1003)}
        data['choice'] = {'operation': 'Choice', 'training_category': 'receiver'}
        data['other'] = {'operation': 'Score', 'training_category': 'reader'}
        release = SimpleNamespace(purpose='training', record=data.__getitem__,
            manifest={'records':[{'id':k, 'selected':True} for k in data]})
        ids = training_ids(release)
        self.assertEqual(set(ids), {str(i) for i in range(1003)})
        counts = Counter(k for i in range(1,2001) for k in visit_ids(ids,i))
        self.assertEqual(sum(counts.values()),16000)
        self.assertEqual(Counter(counts.values()), {16:955,15:48})
        for purpose in ('development','locked'):
            release.purpose=purpose
            with self.assertRaises(ValueError): training_ids(release)
        for window in (0,2001):
            with self.assertRaises(ValueError): visit_ids(ids,window)

    def test_choice_bypass_and_reason_preservation_and_scope_restore(self):
        old = {'feature':object(), 'reason_slots':object(), 'status_logits':'old', 'raw':0, 'value':0}
        original = lambda *args: dict(old)
        codec = SimpleNamespace(evaluate=original)
        with patch('qwen_one_two.training.receiver_repair.readout',return_value={'status_logits':'new','raw':1,'value':1}) as fn:
            with self.assertRaisesRegex(RuntimeError,'stop'):
                with installed(codec,object()):
                    self.assertEqual(codec.evaluate(object(),Port.CHOICE),old)
                    fn.assert_not_called()
                    for port in (Port.NOUL,Port.SCORE):
                        result=codec.evaluate(object(),port)
                        self.assertIs(result['feature'],old['feature'])
                        self.assertIs(result['reason_slots'],old['reason_slots'])
                        self.assertEqual(result['status_logits'],'new')
                    raise RuntimeError('stop')
        self.assertIs(codec.evaluate,original)

    def test_import_does_not_load_models(self):
        code = """
import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','safetensors','numpy'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import scripts.train_receiver_repair
"""
        subprocess.run([sys.executable,'-c',code],check=True)


if __name__ == '__main__': unittest.main()
