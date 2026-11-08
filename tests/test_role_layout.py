"""Model-free RD-04 contract arithmetic and early rejection."""
import unittest
from qwen_one_two.codec.schema import Kind,Port,Invalid
from qwen_one_two.codec.layout import *


class LayoutTests(unittest.TestCase):
    def test_legacy_lengths_and_no_conversion(self):
        self.assertEqual(semantic_lengths(Kind.QUESTION,LEGACY),(8,))
        with self.assertRaises(Invalid):semantic_lengths(Kind.QUESTION,LEGACY,(8,))
        with self.assertRaises(Invalid):semantic_lengths(Kind.QUESTION,DIRECT)

    def test_all_return_statuses_and_largest_candidate_are_reserved(self):
        self.assertEqual(admit(packet_costs(Port.NOUL,8,2))['return_max'],50)
        score=admit(packet_costs(Port.SCORE,8,2,rubric=(2,2,2,2,0)))
        self.assertEqual((score['request'],score['return_max']),(22,59))
        choice=admit(packet_costs(Port.CHOICE,4,3,candidates=((2,0),(32,15))))
        self.assertEqual(choice['return_max'],58)
        self.assertEqual(choice['assessed_return'],58)
        self.assertEqual(choice['candidate_ends'][-1],choice['request']-2)

    def test_existing_envelopes_and_complete_fields_fail_closed(self):
        with self.assertRaisesRegex(Invalid,'64 positions'):admit(packet_costs(Port.NOUL,16,12))
        with self.assertRaisesRegex(Invalid,'256 positions'):
            admit(packet_costs(Port.CHOICE,1,1,candidates=((1,0),)*16,evidence=(33,)*8))
        for length in (0,33,True):
            with self.assertRaises(Invalid):semantic_lengths(Kind.QUESTION,DIRECT,(length,))
        with self.assertRaises(Invalid):semantic_lengths(Kind.RUBRIC,DIRECT,(1,1,1,1,0),midpoint=True)

    def test_exact_attribute_cost_and_midpoint_presence(self):
        self.assertEqual(attribute_cost(LEGACY,2,16),19)
        self.assertEqual(attribute_cost(DIRECT,32,16),49)
        self.assertEqual(semantic_lengths(Kind.RUBRIC,DIRECT,(1,2,3,4,0)),(1,2,3,4,0))


if __name__=='__main__':unittest.main()
