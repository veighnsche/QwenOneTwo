"""RD-03 model-free policy/lineage tests. No model or tensor forward fixtures."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from qwen_one_two.training.frozen_source import (
    D_SOURCE,FROZEN_SOURCE,GPU_CHECKS,fingerprint,source_key,source_mode,validate_arithmetic)
from qwen_one_two.training.inputs import canonical,digest,Source,Binding,Node,Provenance
from qwen_one_two.training.clean import CleanView,tokenize_excerpt


def identity():
    return fingerprint(base='1'*64,live_w='2'*64,tokenizer='3'*64,
        configuration={name:'4'*64 for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')},
        arithmetic={'feedback':'fp32','fixture':'metadata only'},implementation={'fixture.py':'5'*64})


def definition(text,name):
    return next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)


class CharacterTokenizer:
    """Literal character/offset plumbing only; never a learned tokenizer/model."""
    def apply_chat_template(self,messages,**kwargs):return '['+messages[0]['content']+']'
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}


class FrozenSourceTests(unittest.TestCase):
    def test_metadata_and_gpu_test_collection_have_no_model_imports(self):
        program='''
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers','peft'}:
   raise RuntimeError('model import forbidden: '+name)
sys.meta_path.insert(0,Block())
import qwen_one_two.training.frozen_source
import qwen_one_two.training.clean
import qwen_one_two.training.receiver_plan
import tests.test_frozen_source_gpu
'''
        subprocess.run([sys.executable,'-c',program],check=True)

    def test_explicit_offline_policy_and_mutable_control(self):
        self.assertEqual(source_mode(D_SOURCE),'D')
        self.assertEqual(source_mode(FROZEN_SOURCE,cache=True),'W')
        for policy,cache,regime in [(D_SOURCE,True,'offline_component'),('typo',False,'offline_component'),
                                    (FROZEN_SOURCE,False,'native')]:
            with self.assertRaises(ValueError):source_mode(policy,cache=cache,regime=regime)

    def test_v3_admission_rejects_failed_or_incomplete_profiles(self):
        fields=dict(compensated=True,fp32_residual=True,feedback_precision='fp32',precision_parts=2,
                    temporal_chunk=32,recurrence_backend='graph-v1',attention_block_backend='eager',round_reference_inputs=False)
        validate_arithmetic(SimpleNamespace(**fields))
        for key,value in [('compensated',False),('fp32_residual',False),('feedback_precision','bf16'),
                          ('precision_parts',3),('temporal_chunk',16),('recurrence_backend','unknown'),
                          ('round_reference_inputs',True)]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'v3 arithmetic'):
                validate_arithmetic(SimpleNamespace(**dict(fields,**{key:value})))

    def test_source_key_binds_complete_bytes_tokenization_mapping_and_dependencies(self):
        original=identity();key=source_key(original,b'ab',[91,97,98,93],[1,2],((0,1),(1,2)))
        self.assertEqual(key,source_key(deepcopy(original),b'ab',[91,97,98,93],[1,2],((0,1),(1,2))))
        self.assertNotEqual(key,source_key(original,b'ac',[91,97,99,93],[1,2],((0,1),(1,2))))
        self.assertNotEqual(key,source_key(original,b'ab',[92,97,98,94],[1,2],((0,1),(1,2))))
        for name in ('base','live_w','tokenizer'):
            changed=deepcopy(original);changed[name]='6'*64
            self.assertNotEqual(key,source_key(changed,b'ab',[91,97,98,93],[1,2],((0,1),(1,2))))
        for name in ('configuration','implementation','arithmetic'):
            changed=deepcopy(original)
            first=next(iter(changed[name]));changed[name][first]='7'*64
            self.assertNotEqual(key,source_key(changed,b'ab',[91,97,98,93],[1,2],((0,1),(1,2))))

    def test_no_receiver_or_D_hash_in_source_dependencies(self):
        original=identity()
        for field in ('checkpoint','D','receiver','reference','native_state'):
            changed=dict(original,**{field:'8'*64})
            with self.assertRaisesRegex(ValueError,'unexpected source cache dependency'):
                source_key(changed,b'ab',[97,98],[0,1],((0,1),(1,2)))

    def test_missing_fingerprints_and_incomplete_or_ambiguous_mappings_reject(self):
        incomplete=identity();incomplete['configuration'].pop('chat_template.jinja')
        with self.assertRaisesRegex(ValueError,'complete source tokenizer'):
            source_key(incomplete,b'a',[97],[0],((0,1),))
        for ids,selected,offsets in [([97,98],[0],((0,1),)),([97,98],[0,1],((0,1),(0,2))),
                                     ([97,98],[0,2],((0,1),(1,2))),([97]*513,[0,1],((0,1),(1,2))),
                                     ([True,98],[0,1],((0,1),(1,2)))]:
            with self.subTest(ids=len(ids),selected=selected),self.assertRaises(ValueError):
                source_key(identity(),b'ab',ids,selected,offsets)
        with self.assertRaises(UnicodeDecodeError):
            source_key(identity(),'α'.encode(),[1,2],[0,1],((0,1),(1,2)))

    def test_tokenization_retains_all_permitted_bytes_and_rejects_expansion(self):
        body='aαb'.encode();ids,selected,offsets=tokenize_excerpt(CharacterTokenizer(),body)
        self.assertEqual(offsets,((0,1),(1,3),(3,4)))
        self.assertEqual(len(selected),3)
        self.assertEqual(source_key(identity(),body,ids,selected,offsets),source_key(identity(),body,ids,selected,offsets))
        class Cropped(CharacterTokenizer):
            def __call__(self,text,**kwargs):
                result=super().__call__(text,**kwargs);result['offset_mapping'][1]=(0,2);return result
        with self.assertRaisesRegex(ValueError,'boundary'):tokenize_excerpt(Cropped(),body)

    def test_hidden_parent_information_stays_outside_model_view_and_source_cache(self):
        def view(hidden):
            text=b'ab'+hidden;source=Source('parent',1,text,digest(text));binding=Binding('parent',1,0,2)
            p=Provenance({'fact':Node('source_fact',digest(b'ab'),binding=binding)},{'parent':source})
            return CleanView((),(binding,),frozenset({'fact'})),p
        a,p=view(b' hidden true');b,q=view(b' hidden false and longer')
        self.assertEqual(a.materialize(p),b.materialize(q))
        self.assertNotEqual(p.sources['parent'].sha256,q.sources['parent'].sha256)
        keys=[]
        for v,provenance in [(a,p),(b,q)]:
            excerpt=v.materialize(provenance)[0][0]
            ids,selected,offsets=tokenize_excerpt(CharacterTokenizer(),excerpt)
            keys.append(source_key(identity(),excerpt,ids,selected,offsets))
        self.assertEqual(*keys)

    def test_expanded_W_branch_has_no_adapter_arithmetic_or_mutation(self):
        text=Path('qwen_one_two/codec/backbone.py').read_text();fn=definition(text,'expanded_dual')
        branch=fn.body[0];self.assertIsInstance(branch,ast.If)
        code=ast.unparse(branch)
        self.assertIn("MODE.get() == 'W'",code);self.assertIn('return feedback_linear(',code)
        for forbidden in ('R_A','R_B','D_A','D_B','requires_grad_','functional_call','.data','copy_','zero_'):
            self.assertNotIn(forbidden,code)
        old=definition(subprocess.check_output(['git','show','850a2da:qwen_one_two/codec/backbone.py'],text=True),'expanded_dual')
        self.assertEqual([ast.dump(n) for n in fn.body[1:]],[ast.dump(n) for n in old.body])

    def test_native_layer_cache_graph_and_permitted_view_builders_are_unchanged(self):
        old=subprocess.check_output(['git','show','850a2da:qwen_one_two/g1_backbone.py'],text=True)
        before=definition(old,'Backbone');after=definition(Path('qwen_one_two/g1_backbone.py').read_text(),'Backbone')
        for name in ('_layer','initial','embedding','snapshot_state'):
            self.assertEqual(ast.dump(next(n for n in before.body if getattr(n,'name',None)==name)),
                             ast.dump(next(n for n in after.body if getattr(n,'name',None)==name)))
        fn=next(n for n in after.body if getattr(n,'name',None)=='forward')
        self.assertEqual(fn.args.kwonlyargs[0].arg,'offline_source')
        self.assertIs(fn.args.kw_defaults[0].value,False)
        self.assertIn('W is exclusively an explicit offline source forward',ast.unparse(fn))
        for path in ('qwen_one_two/codec/session.py','qwen_one_two/training/producer.py','qwen_one_two/training/binding.py'):
            self.assertEqual(Path(path).read_bytes(),subprocess.check_output(['git','show','850a2da:'+path]))
        old=definition(subprocess.check_output(['git','show','850a2da:qwen_one_two/training/clean.py'],text=True),'CleanView')
        current=definition(Path('qwen_one_two/training/clean.py').read_text(),'CleanView')
        self.assertEqual(ast.dump(old),ast.dump(current))

    def test_GPU_checks_are_implemented_registered_and_pending_not_executed(self):
        tests=definition(Path('tests/test_frozen_source_gpu.py').read_text(),'FrozenSourceGPU')
        methods={n.name for n in tests.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(GPU_CHECKS.values()),{name for name in methods if name.startswith('test_')})
        setup=ast.unparse(next(n for n in tests.body if getattr(n,'name',None)=='setUpClass'))
        self.assertLess(setup.index('QOT_RD03_GPU_QUALIFY'),setup.index('import torch'))
        self.assertLess(setup.index('CUDA_VISIBLE_DEVICES'),setup.index('import torch'))
        self.assertIn('explicit RD-07 authorization',setup)


if __name__=='__main__':unittest.main()
