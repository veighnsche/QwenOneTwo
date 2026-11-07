"""Model-free checks of the fixed diagnostic source transform, not training data."""
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from qwen_one_two.training.numeric_access import count_specs,byte_span,NumericConstruction,legacy_hash
from qwen_one_two.training.inputs import Source,Binding,Node,Provenance,digest,canonical
from qwen_one_two.training.clean import CleanView


def fixture(required='200',completed='173'):
    texts=[f'The requirement is {required} required checkpoints.'.encode(),f'{completed} required checkpoints are complete.'.encode()]
    sources={str(i):Source(str(i),1,t,digest(t)) for i,t in enumerate(texts)}
    bindings=tuple(Binding(str(i),1,0,len(t)) for i,t in enumerate(texts))
    nodes={str(i):Node('source_fact',digest(t),binding=bindings[i]) for i,t in enumerate(texts)}
    return CleanView((),bindings,frozenset(nodes),operation='Score',rubric_direction=1),Provenance(nodes,sources)


class NumericAccessTests(unittest.TestCase):
    def test_key_meanings_do_not_contain_values_and_arms_share_view(self):
        v,p=fixture();a=NumericConstruction('literal');b=NumericConstruction('int64')
        av,ap,_=a.prepare(v,p,{});bv,bp,_=b.prepare(v,p,{})
        self.assertEqual(av.materialize(ap),bv.materialize(bp))
        self.assertNotEqual(a.receipt['input_sha256'],b.receipt['input_sha256'])
        for ref in av.references:
            self.assertNotIn(b'173',ref.content);self.assertNotIn(b'200',ref.content)
        changed,q=fixture('20','5');cv,_,_=a.prepare(changed,q,{})
        self.assertEqual([r.content for r in cv.references],[r.content for r in av.references])
        self.assertEqual([json.loads(r.content)['fields']['property'] for r in av.references],['requirement','complete'])

    def test_direction_migration_is_explicit_but_v2_hash_differs(self):
        v,p=fixture();body=v.materialize(p)[1];lower=replace(v,rubric_direction=2).materialize(p)[1]
        self.assertNotEqual(digest(body),digest(lower))
        self.assertEqual(legacy_hash(body),legacy_hash(lower))
        with self.assertRaises((ValueError,KeyError)):legacy_hash(canonical({'schema':'clean-view-v1'}))

    def test_bad_units_templates_lengths_and_spans_fail(self):
        for texts in ((b'The requirement is 20 panels.',b'5 seats are complete.'),
                      (b'The requirement is 20 panels.',b'Unknown panels are complete.'),
                      (b'The requirement is 123456789 panels.',b'5 panels are complete.')):
            with self.assertRaises(ValueError):count_specs(texts)
        self.assertEqual(byte_span(((0,3),(3,5),(5,9)),2,7),(0,3,2,2))
        with self.assertRaises(ValueError):byte_span(((0,3),),2,4)

    def test_metadata_import_does_not_load_models(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import scripts.compare_numeric_access
"""],check=True)


if __name__=='__main__':unittest.main()
