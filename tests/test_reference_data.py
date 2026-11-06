"""Schema-only role-view and schedule tests; no real reviews or model work."""
import json
import tempfile
import unittest
from pathlib import Path
from release_fixtures import row,write_release
from qwen_one_two.training.release import load_release
from qwen_one_two.training.inputs import SLOTS
from qwen_one_two.training.reference_data import catalogue,ViewSampler


class ReferenceDataTests(unittest.TestCase):
    def release(self,split='train'):
        meanings={'question':{'proposition':'SCHEMA ONLY proposition'},'target':{'referent':'SCHEMA ONLY referent'},
            'alternative':{'identity':'SCHEMA ONLY A','properties':'SCHEMA ONLY properties'},
            'key':{'property':'SCHEMA ONLY property'},'reason':{'status':'Insufficient','support':'SCHEMA ONLY missing fact'}}
        rows=[row(str(i),split=split,training_category='reference',operation='Noul',reference_role=role,
                  model_inputs={'role':role,'meaning':meanings.get(role,'SCHEMA ONLY '+role)})
              for i,role in enumerate(SLOTS)]
        rows.append(row('writer',split=split,training_category='writer',operation='Choice',
            supervision={'request_semantics':{'question':{'proposition':'SCHEMA ONLY reused proposition'},
                'target':{'referent':'SCHEMA ONLY target'},'alternatives':[{'identity':'A','properties':'SCHEMA ONLY A'}]},
                'result':{'winner':'DO NOT ENCODE'},'continuations':['DO NOT ENCODE'],
                'reason':{'status':'NoFeasibleChoice','support':'SCHEMA ONLY constraints'}},
            labels={'schema_version':'test-only-v1','outcome':'DO NOT ENCODE'},guidance={'preferred_effort':'DO NOT ENCODE'}))
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        write_release(Path(temp.name),rows,purpose='training' if split=='train' else 'development')
        return load_release(temp.name)

    def test_reuse_is_whitelisted_hash_bound_and_reason_is_separate(self):
        release=self.release();views=catalogue(release)
        self.assertEqual({v.role for v in views},set(SLOTS))
        for view in views:
            self.assertNotIn(b'DO NOT ENCODE',view.content)
            self.assertEqual(json.loads(view.content)['role'],view.role)
            for key,sha,path in view.sources:
                from qwen_one_two.training.inputs import canonical,digest
                self.assertEqual(sha,digest(canonical(release.record(key))))
                if view.role!='reason':self.assertNotEqual(path,('supervision','reason'))
        reason=next(v for v in views if b'NoFeasibleChoice' in v.content)
        self.assertEqual(reason.role,'reason');self.assertEqual(reason.operation,'Choice')

    def test_development_is_never_a_fitting_catalogue(self):
        with self.assertRaisesRegex(ValueError,'training release'):catalogue(self.release('development'))

    def test_schedule_covers_all_views_and_resume_is_exact(self):
        release=self.release();views=catalogue(release)
        sampler=ViewSampler(views,release.identity);seen=set()
        for _ in range(len(views)):seen.update(sampler.next_window())
        self.assertEqual(seen,{v.identity for v in views})
        resumed=ViewSampler(views,release.identity);resumed.load_state_dict(sampler.state_dict())
        self.assertEqual(sampler.next_window(),resumed.next_window())
        bad=resumed.state_dict();bad['role_cursor']+=1
        with self.assertRaisesRegex(ValueError,'drift'):sampler.load_state_dict(bad)
