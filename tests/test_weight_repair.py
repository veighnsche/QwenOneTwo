"""Metadata-only schema fixtures, never actual reviewed calibration data."""
from types import SimpleNamespace
import unittest
from qwen_one_two.training.inputs import SLOTS,digest
from qwen_one_two.training.weight_repair import calibration_views


class CalibrationTests(unittest.TestCase):
    def test_selection_is_order_invariant_unique_and_role_bounded(self):
        views=[SimpleNamespace(identity=digest(f'SCHEMA ONLY:{role}:{i}'.encode()),role=role)
               for role in SLOTS for i in range(12 if role!='key' else 3)]
        a=calibration_views(views);b=calibration_views(reversed(views))
        self.assertEqual(a,b);self.assertEqual(len(a),75)
        self.assertEqual(len({v.identity for v in a}),75)
        for role in SLOTS:self.assertEqual(sum(v.role==role for v in a),3 if role=='key' else 8)

    def test_missing_role_and_duplicate_identity_reject(self):
        views=[SimpleNamespace(identity=digest(role.encode()),role=role) for role in SLOTS]
        with self.assertRaisesRegex(ValueError,'coverage'):calibration_views(views[:-1])
        with self.assertRaisesRegex(ValueError,'duplicate'):calibration_views(views+[views[0]])
