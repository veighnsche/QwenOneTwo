"""Model-free ABI/resource fixtures; no neural computation on CPU."""
import unittest
from dataclasses import replace
from qwen_one_two.codec.schema import *
from qwen_one_two.codec.runtime import Runtime


class ExactABI(unittest.TestCase):
    def test_signed_and_decimal_roundtrip(self):
        for n in (-(1<<63),-123,-1,0,1,120,(1<<63)-1):
            with self.subTest(n=n):
                atoms=encode_atoms(ValueKind.INT64,n)
                self.assertEqual(decode_atoms(ValueKind.INT64,atoms),n)
                self.assertEqual(atoms[0].data,n.to_bytes(8,'big',signed=True))
                for e in (-18,-2,0,18):
                    a=encode_atoms(ValueKind.DECIMAL,(n,e))
                    self.assertEqual(a[1].data,bytes(7)+bytes([e&255]))
                    self.assertEqual(decode_atoms(ValueKind.DECIMAL,a),(n,e))
        for n in (-(1<<63)-1,1<<63,True):
            with self.assertRaises(Invalid):encode_atoms(ValueKind.INT64,n)

    def test_booleans_handles_and_utf8_boundaries(self):
        for value in (False,True):
            self.assertIs(decode_atoms(ValueKind.BOOLEAN,encode_atoms(ValueKind.BOOLEAN,value)),value)
        for h in (0,1,255):
            self.assertEqual(decode_atoms(ValueKind.REFERENCE,encode_atoms(ValueKind.REFERENCE,None,h)),h)
        for value in (b'',b'a',b'12345678',b'123456789',('é'*64).encode(),('abc🙂').encode(),('1234567🙂').encode()):
            atoms=encode_atoms(ValueKind.UTF8,value)
            self.assertEqual(len(atoms),(len(value)+7)//8)
            self.assertEqual(decode_atoms(ValueKind.UTF8,atoms,len(value)),value)
        for value in (b'\xff',b'\xc3',b'a'*129):
            with self.assertRaises(Invalid):encode_atoms(ValueKind.UTF8,value)
        with self.assertRaises(Invalid):decode_atoms(ValueKind.UTF8,(Atom(Role.UTF8,b'a\x01'+bytes(6)),),1)

    def test_numeric_parser_only_accepts_selected_whole_literal(self):
        self.assertEqual(parse_numeric(b'1.20',ValueKind.DECIMAL),(120,-2))
        self.assertEqual(parse_numeric(b'-12.00e+2',ValueKind.DECIMAL),(-1200,0))
        self.assertEqual(parse_numeric(b'-9223372036854775808',ValueKind.INT64),-(1<<63))
        for literal in (b'available',b'fee 120',b'120 hours',b' 120',b'NaN',b'1e19'):
            with self.assertRaises(Invalid):parse_numeric(literal,ValueKind.DECIMAL)

    def test_metadata_independent_bit_oracle_and_zero_handle(self):
        md=Metadata(Port.NOUL,kind=Kind.SNAPSHOT,owner=0,question=1,target=2,following=12,complete=13,attempt=1)
        got=md.encode(Role.REQUEST)
        word=int(''.join(map(str,got)),2)
        def field(a,n):return (word>>(128-a-n))&((1<<n)-1)
        for a,n,v in ((0,4,1),(4,3,1),(15,4,9),(20,1,1),(25,1,1),(26,1,1),
                      (30,1,0),(31,1,1),(32,8,0),(48,9,12),(64,8,1),(72,8,2),(114,9,13)):
            self.assertEqual(field(a,n),v,(a,n))
        with self.assertRaises(Invalid):replace(md,owner=None).encode(Role.REQUEST)
        attr=Metadata(Port.NOUL,kind=Kind.TARGET,owner=0,following=2,value_kind=ValueKind.BOOLEAN,present=False)
        self.assertNotEqual(attr.encode(Role.ATTRIBUTE_DESC),replace(attr,unit=0).encode(Role.ATTRIBUTE_DESC))
        self.assertTrue(all(x==0 for x in got[125:]))
        with self.assertRaises(Invalid):replace(md,primary=3).encode(Role.REQUEST)
        with self.assertRaises(Invalid):replace(md,complete=257,following=256).encode(Role.REQUEST)
        with self.assertRaises(Invalid):replace(md,direction=3).encode(Role.REQUEST)
        with self.assertRaises(ValueError):md.encode(41)

    def test_control_budget_and_attempt_caps_without_model(self):
        rt=Runtime('pure',16,128,16);rt.startup();rt.begin(Port.NOUL)
        for _ in range(32):rt.writer_position()
        with self.assertRaises(Invalid):rt.writer_position()
        rt.writer_action(0)
        self.assertEqual(rt.call.state,'WriterLimit')
        rt.notice()
        with self.assertRaises(Invalid):rt.notice()
        for _ in range(15):
            rt.begin(Port.NOUL);rt.end_attempt('WriterAborted',12)
        with self.assertRaises(Invalid):rt.begin(Port.NOUL)
        self.assertEqual(rt.attempts,16)
        self.assertEqual(rt.b,32)
        rt.close();rt.native();rt.terminate('Completed')
        self.assertFalse(rt.terminate('Cancelled'))
        with self.assertRaises(Invalid):rt.native()
        self.assertEqual(rt.summary()['held_positions'],0)

    def test_zero_budget_does_not_fund_calls(self):
        rt=Runtime('zero',16,0,4);rt.startup()
        with self.assertRaises(Invalid):rt.begin(Port.SCORE)
        rt.native();self.assertEqual((rt.phase,rt.b,rt.a,rt.h),(1,0,1,5))
        rt.terminate('Limited');self.assertEqual(rt.h,6)
        with self.assertRaises(Invalid):Runtime('bad',262000,128,128)


if __name__=='__main__':unittest.main()
