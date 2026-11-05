"""Model-free evidence coverage checks; never execute a model on CPU."""
import itertools
import unittest
from qwen_one_two.codec.qualify_matrix import PROTOCOLS, validate_coverage


class CoverageTests(unittest.TestCase):
    def rows(self, protocol):
        return [dict(zip(('length','port','profile','seed','repeat'),key),comparison={'pass':True})
                for key in itertools.product(PROTOCOLS[protocol],(1,2,3),
                    ('functional','checkpoint','serving','fp32'),(17,29,43),range(3))]

    def test_distinct_protocols(self):
        for name,count in (('v3-512',324),('v3-1024-v1',108)):
            rows=self.rows(name)
            self.assertEqual(len(rows),count)
            validate_coverage(rows,name)
        with self.assertRaises(ValueError):validate_coverage(self.rows('v3-512'),'v3-1024-v1')

    def test_missing_duplicate_failed_or_foreign_case_rejected(self):
        rows=self.rows('v3-1024-v1')
        for altered in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]],
                        [dict(rows[0],length=512)]+rows[1:],
                        [dict(rows[0],comparison={'pass':False})]+rows[1:]):
            with self.assertRaises(ValueError):validate_coverage(altered,'v3-1024-v1')
