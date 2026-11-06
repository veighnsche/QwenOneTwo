import unittest
from qwen_one_two.training.reference_exposure import reallocate


class ExposureTest(unittest.TestCase):
    def test_preserves_firsts_coverage_roles_and_requested_counts(self):
        windows = [['a', 'b', 'x', 'y', 'c', 'd', 'x', 'y']]*10
        roles = dict(a='question', b='question', c='question', d='question', x='key', y='key')
        treatment, audit = reallocate(windows, roles, {'a': 15, 'b': 15})
        self.assertEqual(len(audit['changes']), 10)
        self.assertEqual(audit['after']['a'], 15)
        self.assertEqual(audit['after']['b'], 15)
        self.assertEqual(treatment[0], windows[0])
        self.assertEqual(set(audit['before']), set(audit['after']))
        self.assertTrue(all(roles[a] == roles[b] for w, v in zip(windows, treatment) for a, b in zip(w, v)))
        self.assertTrue(all(v[2:4] == ['x', 'y'] and v[6:] == ['x', 'y'] for v in treatment))
        self.assertEqual((treatment, audit), reallocate(windows, roles, {'a': 15, 'b': 15}))
        self.assertTrue(all(c['removed'] in ('c', 'd') for c in audit['changes']))

    def test_rejects_impossible_and_foreign_allocations(self):
        windows = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
        roles = {k: 'question' for k in windows[0]}
        for desired in ({'a': 2}, {'a': 0}, {'missing': 2}):
            with self.assertRaises(ValueError):
                reallocate(windows, roles, desired)

    def test_unchanged_when_no_additions(self):
        windows = [['a']*8]
        treatment, audit = reallocate(windows, {'a': 'key'}, {'a': 8})
        self.assertEqual(treatment, windows)
        self.assertEqual(audit['changes'], [])


if __name__ == '__main__':
    unittest.main()
