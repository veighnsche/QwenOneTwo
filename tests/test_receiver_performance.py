"""Literal-byte and policy tests. No tensor/model execution or GPU imports."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch
from qwen_one_two.training.host_metrics import (
    packed_fingerprint,fingerprint_transfers,fingerprint_batch_bytes)
from qwen_one_two.training.inputs import canonical
from qwen_one_two.training.receiver_performance import (
    LEGACY,RETAINED,policy,require_qualification,execution_scope,EVIDENCE,GPU_CHECKS,require_matching_policy)


class ReceiverPerformanceTests(unittest.TestCase):
    def entries(self):
        # Empty, signed zero, NaN payload and UTF-8 are opaque audit bytes.
        return [(canonical({'name':name,'shape':[len(body)],'dtype':'OPAQUE'}),body,len(body))
                for name,body in [('a',b''),('b',b'\x00\x00\x00\x80'),
                                  ('c',b'\x01\x00\xc0\x7f'),('d','café'.encode()),('e',b'x'*33)]]

    def test_identical_canonical_stream_across_group_boundaries(self):
        entries=self.entries();expected=hashlib.sha256()
        for header,body,_ in entries:expected.update(header);expected.update(body)
        for limit in (1,4,5,8,16,32,128):
            groups=[]
            def read(values):groups.append(values);return b''.join(values)
            self.assertEqual(packed_fingerprint(entries,read,limit),expected.hexdigest())
            self.assertEqual([v for g in groups for v in g],[v for _,v,_ in entries])
            self.assertTrue(all(sum(map(len,g))<=limit or len(g)==1 for g in groups))
        self.assertEqual(packed_fingerprint([],lambda _:self.fail('empty read'),8),hashlib.sha256().hexdigest())

    def test_content_changes_are_detected_without_a_version_cache(self):
        entries=self.entries();read=lambda values:b''.join(values)
        before=packed_fingerprint(entries,read,128)
        changed=list(entries);h,b,n=changed[2];changed[2]=(h,bytes([b[0]^1])+b[1:],n)
        self.assertNotEqual(packed_fingerprint(changed,read,128),before)
        changed=list(entries);h,b,n=changed[2];changed[2]=(h+b' ',b,n)
        self.assertNotEqual(packed_fingerprint(changed,read,128),before)

    def test_bad_readback_sizes_and_failures_reject(self):
        for result in (b'',b'x'*100):
            with self.assertRaisesRegex(ValueError,'byte count'):packed_fingerprint(self.entries(),lambda _:result,128)
        for limit in (0,-1,True,1.5):
            with self.assertRaises(ValueError):packed_fingerprint([],lambda _:b'',limit)
        with self.assertRaises(ValueError):packed_fingerprint([(b'h',b'x',-1)],lambda _:b'',8)
        failure=RuntimeError('copy failed')
        def read(_):raise failure
        with self.assertRaises(RuntimeError) as error:packed_fingerprint(self.entries(),read,8)
        self.assertIs(error.exception,failure)

    def test_nested_transfer_policy_restores_after_failure(self):
        self.assertEqual(fingerprint_batch_bytes(),0)
        with fingerprint_transfers(32):
            self.assertEqual(fingerprint_batch_bytes(),32)
            with self.assertRaises(RuntimeError):
                with fingerprint_transfers(8):raise RuntimeError('fixture')
            self.assertEqual(fingerprint_batch_bytes(),32)
        self.assertEqual(fingerprint_batch_bytes(),0)
        for n in (-1,True):
            with self.assertRaises(ValueError):
                with fingerprint_transfers(n):self.fail('invalid policy entered')

    def test_execution_policy_is_explicit_and_unqualified_candidate_rejects(self):
        self.assertTrue(policy(LEGACY)['activation_checkpointing'])
        p=policy(RETAINED);self.assertFalse(p['activation_checkpointing'])
        self.assertEqual(p['recurrence_cache_entries'],32)
        self.assertEqual(p['episode_graphs_retained'],1)
        self.assertFalse(p['automatic_OOM_fallback'])
        require_qualification({});require_qualification({'execution_policy':policy(LEGACY)})
        with self.assertRaisesRegex(ValueError,'GPU qualification pending'):
            require_qualification({'execution_policy':p})
        p['activation_checkpointing']=True
        with self.assertRaisesRegex(ValueError,'policy drift'):require_qualification({'execution_policy':p})
        with self.assertRaisesRegex(ValueError,'unknown receiver'):policy('typo')

    def test_scope_restores_storage_policy_on_exception_without_model_work(self):
        calls=[];recurrence=SimpleNamespace(CACHE_LIMIT=8,clear_graph_cache=lambda:calls.append('clear'))
        backbone=SimpleNamespace(checkpointing=True)
        import qwen_one_two.codec as codec
        with patch.dict(sys.modules,{'qwen_one_two.codec.recurrence':recurrence}),patch.object(codec,'recurrence',recurrence,create=True):
            with self.assertRaises(RuntimeError):
                with execution_scope(backbone,RETAINED):
                    self.assertFalse(backbone.checkpointing);self.assertEqual(recurrence.CACHE_LIMIT,32)
                    self.assertEqual(fingerprint_batch_bytes(),32*1024*1024)
                    raise RuntimeError('fixture OOM; no automatic retry')
        self.assertTrue(backbone.checkpointing);self.assertEqual(recurrence.CACHE_LIMIT,8)
        self.assertEqual(fingerprint_batch_bytes(),0);self.assertEqual(calls,['clear','clear'])

    def test_qualification_requires_exact_source_and_evidence_bytes(self):
        # Deliberately fake metadata fixture, never model evidence or a fit plan.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/EVIDENCE;path.parent.mkdir(parents=True)
            sources={'TEST_ONLY.py':'1'*64};p=policy(RETAINED)
            evidence={'status':'PASS','execution_policy':p,'checks':{k:'PASS' for k in GPU_CHECKS},'source_hashes':sources}
            path.write_text(json.dumps(evidence))
            contract={'execution_policy':p,'dependencies':{'sources':sources},
                      'required_evidence':{'performance_qualification_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
            require_qualification(contract,root)
            path.write_text(json.dumps(evidence)+' ')
            with self.assertRaisesRegex(ValueError,'dependency drift'):require_qualification(contract,root)
            evidence['checks'][GPU_CHECKS[0]]='PENDING';path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'GPU qualification pending'):require_qualification(contract,root)

    def test_existing_arithmetic_and_optimizer_are_not_replaced(self):
        # The opt-in policy only sets storage flags and the audit-transfer scope.
        tree=ast.parse(Path('qwen_one_two/training/receiver_performance.py').read_text())
        calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(any('autograd' in c or 'optimizer' in c or c.endswith('.step') for c in calls))

    def test_matched_arms_require_identical_execution_policy(self):
        require_matching_policy({},LEGACY)
        require_matching_policy({'execution_policy':policy(RETAINED)},RETAINED)
        with self.assertRaisesRegex(ValueError,'matched peer'):require_matching_policy({},RETAINED)
        with self.assertRaisesRegex(ValueError,'matched peer'):
            require_matching_policy({'execution_policy':policy(RETAINED)},LEGACY)


if __name__=='__main__':unittest.main()
