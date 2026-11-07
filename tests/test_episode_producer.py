"""Input-boundary tests. No model execution; explicit schema release fixtures."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,PropertyMock

from release_fixtures import row, write_release
from qwen_one_two.training.producer import native_plan, PrerequisiteError
from qwen_one_two.training.release import load_release


class ProducerInputs(unittest.TestCase):
    def release(self,category='interleaved',**overrides):
        record=row('episode',split='train',training_category=category)
        prompt=record['human_input']['prompt']
        record.update(model_inputs={'causal_history':prompt,'evidence':{'facts':['Literal observation.'],'rules':[]},'draft':{},'counters':{}},
                      supervision={'request_semantics':{'question':{'proposition':'OFFLINE REQUEST GLOSS'},'target':{'referent':'item'}},
                                   'continuations':['ANSWER SENTINEL']},
                      trajectory={'calls':[{'operation':'Noul','request_target':{'question':{'proposition':'OFFLINE REQUEST GLOSS'},'target':{'referent':'item'}},
                                            'result_target':{'status':'Assessed','payload':{'p_true':1}}}]})
        record.update(overrides)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        write_release(Path(temp.name),[record],purpose='training')
        release=load_release(temp.name)
        return release,release.record('episode')

    def test_annotation_and_outcome_are_absent_from_executable_plan(self):
        release,record=self.release()
        plan=native_plan(release,record,'guided_joint',1)
        self.assertIn('Literal observation.',plan.prompt)
        for sentinel in ('OFFLINE REQUEST GLOSS','ANSWER SENTINEL','result_target','p_true'):
            self.assertNotIn(sentinel,repr(plan))
        self.assertEqual(plan.calls,(('Noul',0,'higher'),))
        self.assertIsNone(plan.target)

    def test_free_and_transition_controls_follow_contiguous_schedule(self):
        release,record=self.release()
        self.assertEqual(native_plan(release,record,'free_joint',1).calls,())
        self.assertIsNone(native_plan(release,record,'free_joint',1).override)
        self.assertEqual(native_plan(release,record,'transition',2000).calls,())
        self.assertEqual(len(native_plan(release,record,'transition',1).calls),1)

    def test_release_drift_and_unexpected_input_reject(self):
        release,record=self.release()
        changed=copy.deepcopy(record);changed['model_inputs']['result']='gold'
        with self.assertRaisesRegex(ValueError,'drift'):native_plan(release,changed,'guided_joint',1)
        release,record=self.release(model_inputs={'causal_history':'x','result':'gold'})
        with self.assertRaisesRegex(ValueError,'whitelist'):native_plan(release,record,'guided_joint',1)

    def test_components_require_separate_clean_reference_path(self):
        for category in ('reference','receiver','reader','decision_replay'):
            release,record=self.release(category)
            with self.assertRaises(PrerequisiteError):native_plan(release,record,'components',1)

    def test_loss_targets_never_become_direct_prompt(self):
        release,record=self.release('direct')
        release,record=self.release('direct',model_inputs={'prompt':record['human_input']['prompt']})
        plan=native_plan(release,record,'guided_joint',1)
        self.assertEqual(plan.target,'ANSWER SENTINEL')
        self.assertNotIn(plan.target,plan.prompt)

    def test_component_role_view_excludes_outcome_and_nonselected_evidence(self):
        from qwen_one_two.training.component_producer import component_view
        semantics={'question':{'proposition':'Which option?'},'target':{'referent':'item'},
                   'alternatives':[{'identity':'A','properties':'Selected literal'},
                                   {'identity':'B','properties':'WITHHELD NONSELECTED FACT'}]}
        bound={'request_semantics':{k:v for k,v in semantics.items() if k!='alternatives'},
               'status':'Assessed','payload':{'acceptable_selection_ids':['A']},
               'selected_contents':[semantics['alternatives'][0]],'reason':None}
        release,record=self.release('reader',operation='Choice',semantics=semantics,
            model_inputs={'bound_return':bound,'permitted_prefix':''})
        view,provenance,indices=component_view(release,record)
        _,content=view.materialize(provenance)
        self.assertNotIn(b'WITHHELD NONSELECTED FACT',content)
        self.assertNotIn(b'acceptable_selection_ids',content)
        self.assertIn(b'Selected literal',content)
        self.assertEqual(indices['identities'],('A','B'))
        self.assertEqual(view.evidence,())

    def test_reason_exception_is_post_decision_only(self):
        from qwen_one_two.training.component_producer import component_view
        meaning={'question':{'proposition':'Is the rule met?'},'target':{'referent':'item'}}
        reason={'status':'Insufficient','support':'The required field is missing.'}
        bound={'request_semantics':meaning,'status':'Insufficient','payload':None,
               'selected_contents':[],'reason':reason}
        release,record=self.release('reader',operation='Noul',
            model_inputs={'bound_return':bound,'permitted_prefix':''})
        view,p,indices=component_view(release,record)
        self.assertEqual(view.references[indices['reason']].role,'reason')
        self.assertNotIn(b'Insufficient',view.references[indices['question']].content)
        reason['status']='Assessed'
        release,record=self.release('reader',operation='Noul',
            model_inputs={'bound_return':bound,'permitted_prefix':''})
        with self.assertRaisesRegex(ValueError,'non-assessed'):component_view(release,record)

    def test_reference_record_selects_only_one_whitelisted_role(self):
        from qwen_one_two.training.component_producer import reference_record
        release,record=self.release('reference',operation='Noul',reference_role='target',
            model_inputs={'role':'target','meaning':{'referent':'Neutral item'}})
        annotation,p=reference_record(release,record)
        self.assertIn(b'Neutral item',annotation.content)
        self.assertNotIn(b'ANSWER SENTINEL',annotation.content)
        self.assertEqual(annotation.role,'target')
        release,record=self.release('reference',operation='Noul',reference_role='target',
            model_inputs={'role':'target','meaning':{'referent':'Neutral item','status':'Assessed'}})
        with self.assertRaisesRegex(ValueError,'whitelist'):reference_record(release,record)

    def test_admission_never_decodes_full_manifest(self):
        from qwen_one_two.training.component_producer import component_view,reference_record
        from qwen_one_two.training.release import ReviewedRelease
        native,nrow=self.release()
        meaning={'question':{'proposition':'SCHEMA ONLY question'},'target':{'referent':'SCHEMA ONLY target'}}
        component,crow=self.release('receiver',operation='Noul',model_inputs={'request':meaning,'evidence':{}})
        reference,rrow=self.release('reference',operation='Noul',reference_role='target',
                                    model_inputs={'role':'target','meaning':{'referent':'SCHEMA ONLY target'}})
        self.assertTrue(native.sampler_rows())
        with patch.object(ReviewedRelease,'manifest',new_callable=PropertyMock,
                          side_effect=AssertionError('full manifest decoded')):
            native_plan(native,nrow,'components',1)
            view,p,_=component_view(component,crow);view.materialize(p)
            reference_record(reference,rrow)

    def test_unselected_and_locked_admission_reject_before_model_import(self):
        from qwen_one_two.training.component_producer import component_view,reference_record
        from qwen_one_two.training.binding import request_targets
        from qwen_one_two.training.producer import EpisodeProducer
        from qwen_one_two.training.release import ReviewedRelease
        # These are schema-only rejection fixtures, never model inputs/data.
        for purpose,selected in (('training',{'kept'}),('locked',{'kept','blocked'})):
            split='train' if purpose=='training' else 'locked'
            rows=[row(k,split=split,training_category='receiver') for k in ('kept','blocked')]
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
            write_release(Path(temp.name),rows,purpose=purpose,selected=selected)
            release=load_release(temp.name);record=release.record('blocked')
            producer=EpisodeProducer.__new__(EpisodeProducer)
            producer.release=release;producer.reference_evidence=None
            with patch.object(ReviewedRelease,'manifest',new_callable=PropertyMock,
                              side_effect=AssertionError('full manifest decoded')):
                for call in (lambda:component_view(release,record),lambda:reference_record(release,record),
                             lambda:native_plan(release,record,'components',1),
                             lambda:request_targets(None,release,record,None),
                             lambda:producer(record,'components',1)):
                    with self.assertRaisesRegex(ValueError,'locked|unselected'):call()


if __name__=='__main__':unittest.main()
