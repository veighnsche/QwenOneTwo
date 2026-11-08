"""Symbolic packet fixtures only. CUDA tensor/graph assertions belong to RD-07."""
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen_one_two.codec.schema import *
from qwen_one_two.codec.layout import LEGACY,DIRECT,packet_costs
from qwen_one_two.codec.records import Registry,Ref,Semantic,Bank,Snapshot,Reason,Attribute,Value,Source,Span
from qwen_one_two.codec.render import Renderer
from qwen_one_two.codec.runtime import Runtime


class SymbolicRegistry(Registry):
    """Bypass neural normalization for literal metadata fixtures, never tensors."""
    def publish(self,kind,record,previous=None):
        if kind in (Kind.QUESTION,Kind.TARGET,Kind.ALTERNATIVE,Kind.RUBRIC,Kind.REASON):
            with patch('qwen_one_two.codec.records.tensor'):self.validate(kind,record)
            ref=Ref(self.run,kind,self._next);self._next+=1;self._records[ref]=record;return ref
        with patch('qwen_one_two.codec.records.tensor'):return super().publish(kind,record,previous)

    def fork(self):
        other=SymbolicRegistry(self.run);other._records=self._records.copy();other._next=self._next;return other

    def attribute(self,*args,**kwargs):
        with patch('qwen_one_two.codec.records.tensor'):return super().attribute(*args,**kwargs)


def semantic(n,profile,attributes=(),*,rubric=False):
    lengths=(1,2,1,2,0) if rubric else (n,)
    return Semantic(tuple('role-'+str(i) for i in range(n)),attributes,direction=1 if rubric else 0,
                    profile=profile,lengths=lengths if profile==DIRECT else ())


def fixture(port,profile=DIRECT,*,q=2,t=3,attrs=()):
    rt=Runtime('SYMBOLIC-ONLY',10,1024,16,profile=profile);rt.registry=SymbolicRegistry(rt.registry.run);rt.startup();rt.begin(port)
    if profile==LEGACY:q,t=8,2
    question=rt.publish(Kind.QUESTION,semantic(q,profile));target=rt.publish(Kind.TARGET,semantic(t,profile))
    candidates=tuple(rt.publish(Kind.ALTERNATIVE,semantic(4+i if profile==DIRECT else 4,profile,attrs))
                     for i in range(2)) if port==Port.CHOICE else ()
    rubric=rt.publish(Kind.RUBRIC,semantic(6 if profile==DIRECT else 10,profile,rubric=True)) if port==Port.SCORE else None
    return rt,rt.seal(question,target,rubric=rubric,candidates=candidates)


def signature(packet):
    return [(int(p.role),p.owner,p.slot,p.attribute,p.payload.encode(p.role) if isinstance(p.payload,Metadata)
             else p.payload.data if isinstance(p.payload,Atom) else p.payload) for p in packet.positions]


class DirectPacketTests(unittest.TestCase):
    def test_all_thirteen_cells_charge_actual_lengths_and_keep_echoes(self):
        seen=set()
        for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
            for status in tuple(Status)[1:]:
                if status==Status.NO_FEASIBLE_CHOICE and port!=Port.CHOICE:continue
                rt,packet=fixture(port);reserved=rt.call.position_hold-len(packet)
                self.assertEqual(packet.profile,DIRECT);self.assertEqual(packet.positions[0].payload.field_lengths,(2,))
                rt.semantic_positions(len(packet));reason=None if status==Status.ASSESSED else Reason(('r',)*4,profile=DIRECT)
                returned=rt.prepare_result(packet.snapshot,status,
                    scalar=.25 if status==Status.ASSESSED and port!=Port.CHOICE else None,
                    selected_index=1 if status==Status.ASSESSED and port==Port.CHOICE else None,reason=reason)
                self.assertLessEqual(len(returned),reserved)
                self.assertEqual(returned.handles[:len(packet.handles)],packet.handles)
                self.assertEqual([p.payload for p in packet.positions if p.role==Role.QUESTION],
                                 [p.payload for p in returned.positions if p.role==Role.QUESTION])
                self.assertFalse(any(p.role==Role.MIDPOINT for p in returned.positions))
                rt.semantic_positions(len(returned),True);self.assertTrue(rt.consumed(packet.snapshot))
                self.assertEqual(rt.summary()['B'],len(packet)+len(returned))
                self.assertEqual(rt.summary()['held_positions'],0)
                self.assertFalse(rt.consumed(packet.snapshot));seen.add((port,status))
        self.assertEqual(len(seen),13)

    def test_metadata_version_two_has_exact_lengths_and_zero_reserved_tail(self):
        rt,packet=fixture(Port.SCORE)
        for position in packet.positions:
            if not isinstance(position.payload,Metadata):continue
            md=position.payload;bits_=md.encode(position.role)
            value=lambda a,n:int(''.join(map(str,bits_[a:a+n])),2)
            self.assertEqual(value(0,4),2)
            if position.role==Role.REQUEST:self.assertEqual(value(57,7),2)
            if position.role==Role.TARGET_DESC:self.assertEqual(value(57,7),3)
            if position.role==Role.RUBRIC_DESC:
                self.assertEqual(tuple(value(96+6*i,6) for i in range(5)),(1,2,1,2,0))
                self.assertEqual(value(126,2),0)

    def test_largest_candidate_and_variable_key_cost_are_bound(self):
        attr=Attribute(('key',)*7,Value(ValueKind.INT64,True,123,Mode.CONSTRUCT),DIRECT)
        rt,packet=fixture(Port.CHOICE,attrs=(attr,))
        self.assertEqual(attr.cost,9)
        expected=packet_costs(Port.CHOICE,2,3,candidates=((4,9),(5,9)))
        self.assertEqual(len(packet),expected['request'])
        self.assertEqual(packet.candidate_ends,tuple(enumerate(expected['candidate_ends'])))
        keys=[p for p in packet.positions if p.role==Role.KEY];self.assertEqual(len(keys),14)
        self.assertTrue(all(p.payload.data==int_bytes(123) for p in packet.positions if p.role==Role.INT64))
        rt.semantic_positions(len(packet))
        returned=rt.prepare_result(packet.snapshot,Status.ASSESSED,selected_index=1)
        self.assertEqual(len([p for p in returned.positions if p.role==Role.KEY]),7)
        self.assertEqual([p.payload.data for p in returned.positions if p.role==Role.INT64],[int_bytes(123)])
        bank=rt.registry.get(rt.registry.get(packet.snapshot).bank)
        self.assertEqual(rt.registry.get(returned.result).selected,bank.candidates[1])

    def test_overlong_returns_reject_atomically_without_dispatch_or_registry_changes(self):
        rt=Runtime('REJECT',10,1024,16,profile=DIRECT);rt.registry=SymbolicRegistry('REJECT');rt.startup();rt.begin(Port.NOUL)
        q=rt.publish(Kind.QUESTION,semantic(16,DIRECT));t=rt.publish(Kind.TARGET,semantic(12,DIRECT))
        before=rt.registry.refs();position=rt.position
        with self.assertRaisesRegex(Invalid,'64 positions'):rt.seal(q,t)
        self.assertEqual(rt.registry.refs(),before);self.assertEqual(rt.dispatched,0);self.assertEqual(rt.position,position)
        self.assertIsNone(rt.call.snapshot)

    def test_mixed_layouts_and_mutation_of_caller_records_do_not_rebind(self):
        rt,packet=fixture(Port.NOUL);snapshot=rt.registry.get(packet.snapshot)
        copy=rt.registry.get(snapshot.question)
        self.assertEqual(copy.lengths,(2,))
        with self.assertRaisesRegex(Invalid,'layout mismatch'):
            rt.registry.publish(Kind.SNAPSHOT,Snapshot(Port.NOUL,snapshot.question,
                rt.registry.publish(Kind.TARGET,semantic(2,LEGACY)),profile=DIRECT))
        with self.assertRaises(Invalid):rt.registry.get(Ref('FOREIGN',Kind.QUESTION,1))

    def test_legacy_renderings_match_the_original_renderer(self):
        text=subprocess.check_output(['git','show','483c256:qwen_one_two/codec/render.py'],text=True)
        scope={'__name__':'qwen_one_two.codec._legacy_fixture','__package__':'qwen_one_two.codec'}
        import sys,types
        module=types.ModuleType(scope['__name__']);module.__dict__.update(scope);sys.modules[scope['__name__']]=module
        try:
            exec(compile(text,'legacy-renderer-fixture','exec'),module.__dict__)
            for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
                rt,packet=fixture(port,LEGACY)
                old=module.Renderer(rt.registry).request(packet.snapshot)
                self.assertEqual(signature(packet),signature(old))
                self.assertEqual(packet.candidate_ends,old.candidate_ends)
                self.assertEqual(Renderer(rt.registry).return_max(packet.snapshot),module.Renderer(rt.registry).return_max(packet.snapshot))
        finally:sys.modules.pop(scope['__name__'],None)


if __name__=='__main__':unittest.main()
