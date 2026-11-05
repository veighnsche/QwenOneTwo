"""Identifier-only scheduling fixtures; not semantic data/reviews."""
import unittest
from qwen_one_two.training.sampler import Sampler
from qwen_one_two.training.profile import schedule, guided_events


class ScheduleTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'{category}-{i}',category=category,split='train')
                for category in ('interleaved','direct','language_replay','decision_replay') for i in range(3)]

    def test_exact_joint_recipe_resume_and_no_silent_replacement(self):
        rows=self.rows();a=Sampler(rows,'guided_joint',release_hash='a'*64)
        first=a.next_window();categories=[a.rows[key]['category'] for key in first]
        self.assertEqual(categories,['interleaved']*4+['direct']*2+['language_replay','decision_replay'])
        b=Sampler(rows,'guided_joint',release_hash='a'*64);b.load_state_dict(a.state_dict())
        for _ in range(4):self.assertEqual(a.next_window(),b.next_window())
        bad=Sampler(rows[:-3],'guided_joint',release_hash='a'*64)
        with self.assertRaises(ValueError):bad.next_window()
        self.assertEqual(bad.windows,0);self.assertFalse(bad.cursors)
        with self.assertRaises(ValueError):Sampler([dict(rows[0],split='locked')],'guided_joint',release_hash='a'*64)

    def test_schedule_boundaries_and_contiguous_guidance(self):
        self.assertEqual(guided_events('guided_joint',1,8),8)
        self.assertEqual(guided_events('transition',1,8),8)
        self.assertEqual(guided_events('transition',2000,8),0)
        self.assertEqual(guided_events('transition',500,8,compatible=False),0)
        self.assertEqual(guided_events('free_joint',1,8),0)
        self.assertAlmostEqual(schedule('reference',1),.02)
        self.assertAlmostEqual(schedule('reference',1000),.1)

    def test_component_recipe_balances_operations_and_assessed_fraction(self):
        from collections import Counter
        from qwen_one_two.training.sampler import NONASSESSED
        rows=[]
        for category in ('writer','receiver'):
            for operation,nonassessed in NONASSESSED.items():
                for status in ('Assessed',)+nonassessed:
                    rows.append(dict(id=f'{category}-{operation}-{status}',category=category,operation=operation,
                                     status=status,domain='identifier-fixture',split='train'))
        for category in ('reader','language_replay','decision_replay'):
            rows.append(dict(id=category,category=category,split='train'))
        sampler=Sampler(rows,'components',release_hash='a'*64)
        counts=Counter()
        for _ in range(12):
            for key in sampler.next_window():
                row=sampler.rows[key]
                if row['category']=='receiver':counts[(row['operation'],row['status']=='Assessed')]+=1
        self.assertEqual(set(counts.values()),{4})

    def test_guidance_stops_at_first_incompatible_event(self):
        from qwen_one_two.training.profile import guidance_prefix
        events=('a','b','c','d')
        self.assertEqual(guidance_prefix('guided_joint',1,events,first_incompatible=2),('a','b'))
        self.assertEqual(guidance_prefix('transition',2000,events),())

    def test_resume_cannot_skip_or_replace_scheduled_exposures(self):
        from copy import deepcopy
        sampler=Sampler(self.rows(),'guided_joint',release_hash='a'*64)
        sampler.next_window(); state=sampler.state_dict()
        for mutate in (lambda s:s['cursors'][0].update(value=999),
                       lambda s:s['cursors'].append({'key':['unknown'],'value':1}),
                       lambda s:s.update(windows=2)):
            bad=deepcopy(state); mutate(bad)
            before=sampler.state_dict()
            with self.assertRaisesRegex(ValueError,'scheduled exposures'):
                sampler.load_state_dict(bad)
            self.assertEqual(sampler.state_dict(),before)
