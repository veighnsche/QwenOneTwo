"""Model-free annotation boundaries/admission and GPU-test registration."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from qwen_one_two.training.inputs import canonical,digest,ReferenceInput,Provenance
from qwen_one_two.training.clean import CleanView
from qwen_one_two.training.direct_roles import preflight,plan_role
from qwen_one_two.codec.layout import GPU_CHECKS


class MetadataTokenizer:
    """Literal byte plumbing with one token per complete annotation, no model."""
    def apply_chat_template(self,messages,**kwargs):return '['+messages[0]['content']+']'
    def __call__(self,text,**kwargs):
        return {'input_ids':[1,2,3],'offset_mapping':[(0,1),(1,len(text)-1),(len(text)-1,len(text))]}


class EncodingTests(unittest.TestCase):
    def test_modules_and_GPU_collection_do_not_import_models(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers','peft'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import qwen_one_two.training.direct_roles
import scripts.audit_direct_roles
import tests.test_direct_roles_gpu
"""],check=True)

    def test_content_mapping_excludes_template_positions_and_preserves_order(self):
        body=canonical({'role':'target','fields':{'referent':'existing input bytes'}})
        ref=ReferenceInput('target',body,(),())
        item=plan_role(MetadataTokenizer(),ref)
        self.assertEqual(item['selected'],[1]);self.assertEqual(item['offsets'],((0,len(body)),))
        self.assertEqual(item['input_sha256'],digest(body))
        with self.assertRaisesRegex(ValueError,'Reason remains a return control'):
            plan_role(MetadataTokenizer(),ReferenceInput('reason',body,(),()))

    def test_preflight_is_before_source_model_forward_and_has_no_pooling(self):
        tree=ast.parse(Path('qwen_one_two/training/clean.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='CleanEncoder')
        code=ast.unparse(next(n for n in cls.body if getattr(n,'name',None)=='encode'))
        self.assertLess(code.index('planned = preflight'),code.index('self._body_states'))
        self.assertIn("payload, key = self._body_states(ref.content",code)
        self.assertIn("'body_states'",code)
        for prohibited in ('mean(', 'adaptive_avg_pool','pooler('):self.assertNotIn(prohibited,code)

    def test_projection_is_zero_delta_FP32_and_only_opt_in(self):
        tree=ast.parse(Path('qwen_one_two/codec/neural.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='DirectRoleProjection')
        code=ast.unparse(cls)
        self.assertIn('torch.zeros(1024, 1024',code);self.assertIn('dtype=torch.float32',code)
        self.assertIn('x.float() + F.linear',code)
        codec=next(n for n in tree.body if getattr(n,'name',None)=='Codec')
        constructor=ast.unparse(next(n for n in codec.body if getattr(n,'name',None)=='__init__'))
        self.assertNotIn('DirectRoleProjection',constructor)
        self.assertNotIn('DirectRoleLengths',constructor)

    def test_registered_GPU_checks_have_real_assertions_and_explicit_opt_in(self):
        cls=next(n for n in ast.parse(Path('tests/test_direct_roles_gpu.py').read_text()).body if getattr(n,'name',None)=='DirectRolesGPU')
        methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(GPU_CHECKS.values()),{n for n in methods if n.startswith('test_')})
        for name in GPU_CHECKS.values():self.assertIn('assert',ast.unparse(methods[name]))
        setup=ast.unparse(methods['setUpClass'])
        self.assertLess(setup.index('QOT_RD04_GPU_QUALIFY'),setup.index('import torch'))
        self.assertLess(setup.index('CUDA_VISIBLE_DEVICES'),setup.index('import torch'))


if __name__=='__main__':unittest.main()
