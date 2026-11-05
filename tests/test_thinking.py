"""Pure validation and token-policy checks; no model workloads."""
import unittest
from qwen_one_two.thinking import bounded_int, bounded_score, prompt_text, run_id, score_to_budget

class BudgetTests(unittest.TestCase):
    def test_continuous_score_mapping_has_no_choice_bins(self):
        self.assertEqual(score_to_budget(0), 0)
        self.assertEqual(score_to_budget(1), 131072)
        self.assertEqual(score_to_budget(0.1234), 5495)
        self.assertEqual(score_to_budget(0.124), 5546)
        self.assertEqual(bounded_score(-0.1), 0)
        self.assertEqual(bounded_score(1.1), 1)
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                bounded_score(bad)
        for bad in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                score_to_budget(bad)

    def test_s_curve_is_gentle_at_ends_and_steeper_in_the_middle(self):
        self.assertEqual(score_to_budget(.1), 3670)
        self.assertEqual(score_to_budget(.25), 20480)
        self.assertEqual(score_to_budget(.5), 65536)
        self.assertEqual(score_to_budget(.75), 110592)
        self.assertEqual(score_to_budget(.9), 127402)
        self.assertLess(score_to_budget(.1), .1 * 131072)
        self.assertGreater(score_to_budget(.9), .9 * 131072)
        values = [score_to_budget(i / 100) for i in range(101)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[50] - values[40], values[10] - values[0])

    def test_validation_rejects_bool_float_and_path_ids(self):
        for value in (True, 1.5, -1, 131073):
            with self.assertRaises(ValueError):
                bounded_int(value, 0, 131072, "budget")
        for value in ("../head", "A" * 32, "f" * 33, None):
            with self.assertRaises(ValueError):
                run_id(value)
        for value in (None, "", " " * 20, "a" * 2001):
            with self.assertRaises(ValueError):
                prompt_text({"text": value})


if __name__ == "__main__":
    unittest.main()
