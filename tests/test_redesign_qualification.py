"""Model-free RD-07 execution gates and unchanged arithmetic contracts."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest


class QualificationContracts(unittest.TestCase):
    def test_GPU_module_collection_is_model_free_and_explicitly_guarded(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers','peft'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import tests.test_redesign_qualification_gpu
"""],check=True)
        from tests.test_redesign_qualification_gpu import GPU_CHECKS
        tree=ast.parse(Path('tests/test_redesign_qualification_gpu.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='RedesignQualificationGPU')
        methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(GPU_CHECKS.values()),{n for n in methods if n.startswith('test_')})
        self.assertLess(methods['setUpClass'].index('QOT_RD07_GPU_QUALIFY'),methods['setUpClass'].index('from tests.test_direct_roles_gpu'))
        self.assertIn('exact 3090 UUID',methods['setUpClass'])

    def test_noul_value_uses_the_exposed_raw_node(self):
        tree=ast.parse(Path('qwen_one_two/codec/neural.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='Codec')
        method=next(n for n in cls.body if getattr(n,'name',None)=='evaluate')
        code=ast.unparse(method)
        self.assertIn('raw = logits[1] - logits[0]',code)
        self.assertIn('value=raw.sigmoid(), raw=raw',code)

    def test_long_case_captures_final_state_and_forbids_optimizer_steps(self):
        code=Path('tests/test_redesign_qualification_gpu.py').read_text()
        tree=ast.parse(code);cls=next(n for n in tree.body if getattr(n,'name',None)=='RedesignQualificationGPU')
        trace=ast.unparse(next(n for n in cls.body if getattr(n,'name',None)=='trace'))
        self.assertIn("'end', True",trace);self.assertIn('rt.position, 1024',trace)
        self.assertNotIn('optimizer.step',code)
        compact=code.replace(' ','')
        self.assertIn('1e-4,.99999',compact);self.assertIn('1e-6,1e-5,1e-5',compact)


if __name__=='__main__':unittest.main()
