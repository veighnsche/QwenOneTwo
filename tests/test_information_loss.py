"""Model-free replay and instrumentation lifecycle checks."""
from copy import deepcopy
import subprocess
import sys
import unittest
from qwen_one_two.training.information_loss import validate_replay, CaptureRequest


class InformationLossTests(unittest.TestCase):
    def test_capture_modules_import_without_model_runtime(self):
        code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','safetensors','numpy'}:
            raise AssertionError('model runtime imported: '+fullname)
sys.meta_path.insert(0, Block())
import scripts.diagnose_information_loss
import scripts.audit_architecture
"""
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_replay_rejects_value_input_and_accounting_drift(self):
        saved = {'id': 'fixture', 'record_sha256': 'r', 'view_sha256': 'v', 'actual': {'value': .4},
                 'predicted_value': .4, 'runtime': {'total': 17}, 'failure': None,
                 'score': {'transport_valid': True}}
        validate_replay(deepcopy(saved), saved)
        validate_replay(dict(saved, runtime={'switches': [(1, 'D', 'R')]}),
                        dict(saved, runtime={'switches': [[1, 'D', 'R']]}))
        for key, value in (('predicted_value', .400001), ('view_sha256', 'changed'), ('runtime', {'total': 18})):
            current = deepcopy(saved); current[key] = value
            with self.assertRaisesRegex(ValueError, 'replay drift'):
                validate_replay(current, saved)

    def test_failed_case_cannot_pass_replay(self):
        case = {'failure': 'decode failed', 'score': {'transport_valid': False}}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_replay(case, deepcopy(case))

    def test_capture_restores_methods_on_exception(self):
        class Codec:
            def encode(self, packet):
                raise ValueError('fixture failure')
            def evaluate(self, *args):
                raise AssertionError('unexpected call')
        codec = Codec(); original = codec.encode
        with self.assertRaisesRegex(ValueError, 'fixture failure'):
            with CaptureRequest(codec):
                codec.encode(None)
        self.assertEqual(codec.encode, original)
        self.assertNotIn('encode', vars(codec)); self.assertNotIn('evaluate', vars(codec))


if __name__ == '__main__':
    unittest.main()
