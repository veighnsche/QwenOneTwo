"""Model-free v2 bounds, immutable return bindings and legacy controls."""
import unittest
from tests.test_direct_role_packets import SymbolicRegistry
from qwen_one_two.codec.layout import *
from qwen_one_two.codec.records import Semantic,Reason
from qwen_one_two.codec.runtime import Runtime
from qwen_one_two.codec.schema import Kind,Port,Status,Role,Atom,Metadata,decode_atoms,ValueKind,Invalid


def fixture(profile,port,q=49,t=23):
    rt=Runtime('v2-symbolic',10,1024,16,profile=profile);rt.registry=SymbolicRegistry(rt.registry.run);rt.startup();rt.begin(port)
    def sem(n,**kw):return Semantic(('s',)*n,profile=profile,lengths=(n,) if is_direct(profile) else (),**kw)
    if not is_direct(profile):q,t=8,2
    q=rt.publish(Kind.QUESTION,sem(q));t=rt.publish(Kind.TARGET,sem(t))
    cs=tuple(rt.publish(Kind.ALTERNATIVE,sem(n if is_direct(profile) else 4)) for n in (41,40)) if port==Port.CHOICE else ()
    rubric=rt.publish(Kind.RUBRIC,Semantic(('r',)*67,direction=1,profile=profile,lengths=(11,24,16,16,0))) if port==Port.SCORE and is_direct(profile) else (rt.publish(Kind.RUBRIC,Semantic(('r',)*10,direction=1,profile=profile)) if port==Port.SCORE else None)
    return rt,rt.seal(q,t,rubric=rubric,candidates=cs)


class CompactRoles(unittest.TestCase):
    def test_all_statuses_keep_handle_prefix_and_exact_selection_without_echo(self):
        seen=0
        for profile in COMPACT:
            for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
                for status in tuple(Status)[1:]:
                    if status==Status.NO_FEASIBLE_CHOICE and port!=Port.CHOICE:continue
                    rt,p=fixture(profile,port);self.assertEqual(rt.call.position_hold-len(p),39)
                    rt.semantic_positions(len(p))
                    r=rt.prepare_result(p.snapshot,status,scalar=.25 if port!=Port.CHOICE and status==Status.ASSESSED else None,
                        selected_index=1 if port==Port.CHOICE and status==Status.ASSESSED else None,
                        reason=Reason(('reason',)*4,profile=profile) if status!=Status.ASSESSED else None)
                    self.assertEqual(len(r),3 if status==Status.ASSESSED else 7)
                    self.assertEqual(r.handles[:len(p.handles)],p.handles)
                    self.assertFalse(any(x.role in (Role.QUESTION,Role.TARGET,Role.CANDIDATE) for x in r.positions))
                    self.assertEqual(r.positions[0].payload.question,p.positions[0].payload.question)
                    if port==Port.CHOICE and status==Status.ASSESSED:
                        pos=r.positions[1];self.assertEqual(pos.role,Role.REFERENCE)
                        self.assertEqual(r.handles[decode_atoms(ValueKind.REFERENCE,(pos.payload,))],rt.registry.get(r.result).selected)
                    rt.semantic_positions(len(r),True);self.assertTrue(rt.consumed(p.snapshot));self.assertFalse(rt.consumed(p.snapshot))
                    self.assertEqual(rt.summary()['held_positions'],0);seen+=1
        self.assertEqual(seen,26)

    def test_explicit_capacity_versions_and_no_envelope_increase(self):
        self.assertEqual(semantic_lengths(Kind.QUESTION,COMPACT_DIRECT,(63,)),(63,))
        for profile,n in [(DIRECT,33),(COMPACT_DIRECT,64)]:
            with self.assertRaises(Invalid):semantic_lengths(Kind.QUESTION,profile,(n,))
        self.assertEqual((REQUEST_LIMIT,RETURN_LIMIT),(256,64))
        c=packet_costs(Port.SCORE,49,23,rubric=(11,24,16,16,0),profile=COMPACT_DIRECT)
        self.assertEqual(c['return_max'],39);self.assertEqual(c['request'],143)
        with self.assertRaises(Invalid):admit(dict(c,request=257))
        for profile,version in [(COMPACT_DIRECT,4),(COMPACT_POOLED,3)]:
            _,p=fixture(profile,Port.NOUL)
            self.assertEqual(int(''.join(map(str,p.positions[0].payload.encode(Role.REQUEST)[:4])),2),version)

    def test_cache_stamp_rebind_requires_the_original_frozen_content_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from qwen_one_two.training.clean import CleanEncoder
        parameter=SimpleNamespace(_version=0,requires_grad=False)
        reference=SimpleNamespace(frozen=True,artifact_id='original',parameters=lambda:[parameter],fingerprint=lambda:'original')
        encoder=CleanEncoder.__new__(CleanEncoder);encoder.backbone=object();encoder.reference=reference
        encoder.identity={'reference':'original'};encoder.source_binding={'identity':{'original':'W'}}
        encoder.reference_stamp=((id(parameter),0),)
        with patch('qwen_one_two.training.clean.bind_frozen_source',return_value=encoder.source_binding):
            parameter._version=1;encoder._validate_source()
            self.assertEqual(encoder.reference_stamp,((id(parameter),1),))
            parameter._version=2;reference.fingerprint=lambda:'modified'
            with self.assertRaisesRegex(ValueError,'contents/ownership'):encoder._validate_source()
            self.assertEqual(encoder.reference_stamp,((id(parameter),1),))
            reference.fingerprint=lambda:'original';parameter.requires_grad=True
            with self.assertRaisesRegex(ValueError,'contents/ownership'):encoder._validate_source()
            parameter.requires_grad=False;reference.artifact_id='different'
            with self.assertRaisesRegex(ValueError,'contents/ownership'):encoder._validate_source()


if __name__=='__main__':unittest.main()
