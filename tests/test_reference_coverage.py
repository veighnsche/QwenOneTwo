import unittest
from types import SimpleNamespace
from qwen_one_two.training.inputs import SLOTS
from qwen_one_two.training.reference_data import CompleteViewSampler, coverage_plateau
from qwen_one_two.training.profile import scheduled_rate, PHASES


class CoverageTests(unittest.TestCase):
    def sampler(self):
        views = [SimpleNamespace(identity=str(i), role=tuple(SLOTS)[i%10], operation='Noul',
                                 content=str(i).encode(), sources=()) for i in range(40)]
        return CompleteViewSampler(views, 'a'*64)

    def test_exact_coverage_and_resume_inside_and_between_passes(self):
        sampler = self.sampler(); orders = []
        for _ in range(4):
            seen = []
            for _ in range(sampler.windows_per_pass):
                copy = self.sampler(); copy.load_state_dict(sampler.state_dict())
                ids = sampler.next_window()
                self.assertEqual(copy.next_window(), ids)
                seen.extend(ids)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(set(seen), {str(i) for i in range(40)})
            orders.append(tuple(seen))
        self.assertEqual(len(set(orders)), 4)
        bad = dict(sampler.state_dict(), catalogue_hash='b'*64)
        with self.assertRaisesRegex(ValueError, 'identity'):sampler.load_state_dict(bad)

    def test_only_explicit_reference_profile_extends_the_phase(self):
        self.assertEqual(scheduled_rate('reference', 1001, 'reference-coverage-v1'), 1.)
        for phase, total in PHASES.items():
            with self.assertRaises(ValueError):scheduled_rate(phase, total+1)
            if phase != 'reference':
                with self.assertRaises(ValueError):scheduled_rate(phase, 1, 'reference-coverage-v1')

    def test_plateau_requires_both_metrics_to_stall_across_three_passes(self):
        curves = [dict(panel_exact=20, mean_reconstruction=None),
                  dict(panel_exact=25, mean_reconstruction=.4)]
        for loss in (.41, .42):
            curves.append(dict(panel_exact=24, mean_reconstruction=loss))
            self.assertFalse(coverage_plateau(curves))
        curves.append(dict(panel_exact=25, mean_reconstruction=.4))
        self.assertTrue(coverage_plateau(curves))
        for change in (dict(panel_exact=26, mean_reconstruction=.5),
                       dict(panel_exact=20, mean_reconstruction=.39)):
            self.assertFalse(coverage_plateau(curves+[change]))


if __name__ == '__main__':unittest.main()
