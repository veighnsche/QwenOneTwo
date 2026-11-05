"""Qualification arithmetic/schema fixtures, never measured semantic results."""
from pathlib import Path
import tempfile
import unittest
from release_fixtures import row,write_release
from qwen_one_two.training.release import load_release
from qwen_one_two.training.reference_qualification import suite,relations,summarize,ReferenceEvidence,intervention_relations
from qwen_one_two.training.component_producer import reference_record
from qwen_one_two.training.inputs import SLOTS


class ReferenceQualificationTests(unittest.TestCase):
    def release(self,count=100):
        rows=[]
        for i in range(count):
            role=tuple(SLOTS)[i%10]
            meaning={'status':'Insufficient','support':'SCHEMA ONLY missing field'} if role=='reason' else (
                {'proposition':'SCHEMA ONLY proposition'} if role=='question' else
                {'referent':'SCHEMA ONLY identity'} if role=='target' else
                {'identity':'SCHEMA ONLY alternative'} if role=='alternative' else
                {'property':'SCHEMA ONLY property'} if role=='key' else 'SCHEMA ONLY meaning')
            rows.append(row(str(i),split='development',training_category=None,evaluation_slice='RF',
                reference_role=role,operation='Noul',model_inputs={'role':role,'meaning':meaning}))
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        write_release(Path(temp.name),rows,purpose='development')
        return load_release(temp.name)

    def test_all_development_cases_and_roles_remain_scheduled(self):
        release=self.release()
        self.assertEqual(len(suite(release)),100)
        for key in suite(release):
            item,_=reference_record(release,release.record(key))
            self.assertEqual(item.role,release.record(key)['reference_role'])

    def test_missing_cases_and_unreviewed_contrasts_cannot_disappear(self):
        with self.assertRaisesRegex(ValueError,'100'):suite(self.release(99))
        release=self.release();self.assertEqual(relations(release),())
        cases=[{'recovery':{'exact':True}} for _ in range(100)]
        report=summarize(cases,[],complete=True,unchanged=True)
        self.assertEqual(report['status'],'BLOCKED')
        self.assertIn('contrast',report['blockers'][0])

    def test_thresholds_and_unresolved_denominators(self):
        cases=[{'recovery':{'exact':i<95}} for i in range(100)]
        groups=[{'passed':i<9} for i in range(10)]
        self.assertEqual(summarize(cases,groups,complete=True,unchanged=True)['status'],'PASSED')
        self.assertEqual(summarize(cases[:-1],groups,complete=False,unchanged=True)['status'],'BLOCKED')
        cases[0]['recovery']['exact']=False
        result=summarize(cases,groups,complete=True,unchanged=True)
        self.assertEqual(result['recovery']['unresolved'],6)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(summarize(cases,groups,complete=True,unchanged=False)['status'],'BLOCKED')

    def test_hash_or_frozen_flag_is_not_a_qualification_attestation(self):
        with self.assertRaisesRegex(ValueError,'measured'):
            ReferenceEvidence('a'*64,'b'*64,'c'*64)
        from qwen_one_two.training.component_producer import ComponentProducer
        with self.assertRaisesRegex(ValueError,'measured reference evidence'):
            ComponentProducer(None,None,None,object(),checkpoint_hash=None,profile_hash=None,
                              reference_evidence={'qualified':True},diagnostic_reference=True)

    def intervention_release(self,category,original,changed,*,include_positive=True):
        rows=[]
        for key,side,meaning in (('anchor',0,original),('negative',1,changed)):
            rows.append(row(key,split='development',family='SCHEMA-ONLY-family',semantics=meaning,
                evaluation_slice='X',intervention={'category':category,'side':side,'pair':'SCHEMA-ONLY-pair',
                    'protocol':'offline_semantic_intervention'},construction={'template_version':'controlled-data-v1',
                        'scenario_spec':'SCHEMA ONLY','seed':17,'transformation':'SCHEMA ONLY intervention'}))
        if include_positive:
            rows.append(row('positive',split='development',family='SCHEMA-ONLY-family',
                            evaluation_slice='E',semantics=original))
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        write_release(temp.name,rows,purpose='development')
        return load_release(temp.name)

    def test_reviewed_pair_metadata_and_identical_positive_are_reused(self):
        original={'question':{'proposition':'SCHEMA ONLY P'}}
        changed={'question':{'proposition':'SCHEMA ONLY negation of P'}}
        groups=intervention_relations(self.intervention_release(1,original,changed))
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['positives'],['positive'])
        self.assertEqual(groups[0]['negatives'],['negative'])
        with self.assertRaisesRegex(ValueError,'positive missing'):
            intervention_relations(self.intervention_release(1,original,changed,include_positive=False))

    def test_calibrated_scale_zero_paraphrase_is_not_a_negative(self):
        original={'question':{'proposition':'SCHEMA original score'},
                  'rubric':{'conditions':'SCHEMA divide by requirement','one':'SCHEMA all complete','zero':'SCHEMA none complete'}}
        changed={'question':{'proposition':'SCHEMA calibrated score'},
                 'rubric':{'conditions':'SCHEMA divide by twice requirement','one':'SCHEMA twice complete','zero':'SCHEMA nothing complete'}}
        groups=intervention_relations(self.intervention_release(8,original,changed))
        self.assertEqual({g['role'] for g in groups},{'question','conditions','one'})


if __name__=='__main__':unittest.main()
