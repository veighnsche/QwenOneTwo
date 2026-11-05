"""Model-free checks for proof admission, events and exact metadata bits."""
import unittest
from qwen_one_two.g1_accounting import Ledger, AdmissionError, bit_fields, budget


class G1Bookkeeping(unittest.TestCase):
    def test_exact_fit_and_one_short_do_not_depend_on_predicted_return(self):
        for m in (1,9,16):
            l=Ledger(33,budget=71+m)
            l.attempts=1;l.records=2
            l.commit('startup','H',3);l.commit('thinking','B',4);l.commit('writer','B',3)
            l.admit(14+m,50,4)
            l.commit('request','B',14+m);l.commit('return','B',14)
            l.finish_return(14+m,14)
            self.assertEqual((l.b,l.holds,l.released,l.records),(35+m,0,36,4))
            bad=Ledger(33,budget=70+m,b=7,records=2)
            with self.assertRaises(AdmissionError): bad.admit(14+m,50,4)
            self.assertEqual((bad.dispatched,bad.holds,bad.records),(0,0,3))

    def test_answer_call_spends_original_budget(self):
        l=Ledger(52,budget=128)
        l.commit('thinking','B',4);l.phase=1
        l.commit('answer','A',2);l.commit('call','B',63);l.commit('answer','A',6)
        self.assertEqual((l.b,l.a,l.budget-l.b),(67,8,61))

    def test_duplicate_and_cancel_cannot_create_positions(self):
        l=Ledger(33,b=7,records=2)
        l.admit(23,50,4);l.commit('request','B',23);l.commit('return','B',14)
        l.finish_return(23,14)
        pos=l.position
        with self.assertRaises(AssertionError): l.finish_return(23,14)
        self.assertEqual(l.position,pos)
        l.cancelled=True
        with self.assertRaises(RuntimeError): l.commit('late','H')
        self.assertEqual(l.position,pos)

    def test_metadata_big_endian_and_reserved_zero(self):
        b=bit_fields([(0,4,1),(32,8,255),(48,9,256),(114,9,133)])
        self.assertEqual(b[:4],[0,0,0,1]);self.assertEqual(b[32:40],[1]*8)
        self.assertEqual(b[48:57],[1]+[0]*8);self.assertEqual(b[125:],[0]*3)
        with self.assertRaises(ValueError): bit_fields([(32,8,256)])

    def test_budget_endpoints_and_nonfinite(self):
        self.assertEqual([budget(x) for x in (0,.25,.5,.75,1)],[0,20480,65536,110592,131072])
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError): budget(x)


if __name__=='__main__': unittest.main()
