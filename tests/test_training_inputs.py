"""Model-free adversarial fixtures, not authored/reviewed semantic data."""
import unittest
from qwen_one_two.training.inputs import (Source, Binding, Node, Provenance, Field,
    reference_input, scenario_reference, digest, Exposure, branch_input, validate_record_manifest)
from qwen_one_two.training.clean import CleanView


class InputTests(unittest.TestCase):
    def setUp(self):
        body = 'Orientation complete. Badge expired. α'.encode()
        self.source = Source('original', 1, body, digest(body))
        self.binding = Binding('original', 1, 0, len(b'Orientation complete.'))
        self.nodes = {'fact': Node('source_fact', digest(self.binding.extract({'original':self.source})), binding=self.binding)}

    def field(self, key, text, kind='request_semantics', parents=()):
        self.nodes[key] = Node(kind, digest(text.encode()), parents)
        return Field(text, key)

    def provenance(self):
        return Provenance(self.nodes, {'original':self.source})

    def test_outcomes_never_enter_request_serialization(self):
        fields = {'proposition':self.field('q', 'Does the participant meet access requirements?')}
        row = {'semantics':{'question':{'q':fields}}, 'outcome':False, 'score':0.2, 'effort':0.1}
        first = scenario_reference(row,'question','q',self.provenance())
        row.update(outcome=True, score=0.9, effort=0.8, final_answer='changed', reason='changed')
        self.assertEqual(first, scenario_reference(row,'question','q',self.provenance()))
        fields['winner'] = fields['proposition']
        with self.assertRaises(ValueError):reference_input('question',fields,self.provenance())

    def test_transitive_outcome_and_reason_contamination_rejected(self):
        self.field('label','false','current_outcome')
        f = self.field('q','Neutral looking question',parents=('label',))
        with self.assertRaises(ValueError):reference_input('question',{'proposition':f},self.provenance())
        f = self.field('reason','Missing material information','reason')
        with self.assertRaises(ValueError):reference_input('question',{'proposition':f},self.provenance())

    def test_reason_exception_is_role_and_status_specific(self):
        status = self.field('status','Insufficient','reason')
        support = self.field('support','Required fact unavailable','reason',('fact',))
        value = reference_input('reason',{'status':status,'support':support},self.provenance())
        self.assertEqual(value.role,'reason')
        status = self.field('status','Assessed','reason')
        with self.assertRaises(ValueError):reference_input('reason',{'status':status,'support':support},self.provenance())
        status = self.field('status','NoFeasibleChoice','reason')
        with self.assertRaises(ValueError):reference_input('reason',{'status':status,'support':support},self.provenance(),operation='Noul')

    def test_grounded_source_outcome_is_permitted(self):
        # No keyword censorship: a supplied attributed outcome remains a fact.
        text=b'The source states refund approved.'
        source=Source('s',1,text,digest(text));binding=Binding('s',1,0,len(text))
        p=Provenance({'s':Node('source_fact',digest(text),binding=binding)}, {'s':source})
        reference_input('alternative',{'properties':Field(text.decode(),'s')},p)

    def test_clean_twins_exclude_parent_offsets_and_withheld_bytes(self):
        field=self.field('q','Access requires orientation and an active badge.')
        p=self.provenance();ref=reference_input('question',{'criterion':field},p)
        view=CleanView((ref,),(self.binding,),frozenset({'q','fact'}))
        before=view.materialize(p)
        hidden=b'Orientation complete. Badge active. Other hidden material.'
        self.source=Source('original',1,hidden,digest(hidden))
        after=view.materialize(self.provenance())
        self.assertEqual(before,after)
        self.assertNotIn(b'original',before[1]);self.assertNotIn(b'Badge',before[1])
        bad=CleanView((ref,),(Binding('original',1,22,len(hidden)),),view.permitted_nodes)
        with self.assertRaises(ValueError):bad.materialize(self.provenance())

    def test_coordinates_hash_cycles_and_scope_fail_closed(self):
        p=self.provenance()
        with self.assertRaises(UnicodeDecodeError):Binding('original',1,len(self.source.content)-1,len(self.source.content)).extract(p.sources)
        with self.assertRaises(ValueError):Binding('original',2,0,5).extract(p.sources)
        self.nodes['fact']=Node('source_fact','0'*64,binding=self.binding)
        with self.assertRaises(ValueError):self.provenance()
        self.nodes['fact']=Node('request_semantics','0'*64,('fact',))
        with self.assertRaises(ValueError):self.provenance()

    def test_other_candidate_dependency_rejected_by_view(self):
        f=self.field('other','Another candidate fact')
        own=self.field('own','Own properties',parents=('other',))
        with self.assertRaises(ValueError):reference_input('alternative',{'properties':own},self.provenance(),permitted_nodes={'own'})

    def test_branch_exposure_and_whitelists(self):
        branch_input(Exposure('isolated_reader',packet_origin='reviewed_component',gold_return=True),bound_return='fixture',permitted_prefix='')
        for exposure in (Exposure('joint_reader',gold_return=True), Exposure('joint_receiver',semantic_substitution=True),
                         Exposure('free_evaluation',budget_override=512), Exposure('native_writer',outcome_bearing_prefix=True)):
            with self.assertRaises(ValueError):exposure.validate()
        with self.assertRaises(ValueError):branch_input(Exposure('native_writer'),causal_history='',gold_answer='forbidden')

    def test_release_gate_does_not_approve_drafts(self):
        record=dict(id='synthetic-test',version=1,family='f',split='train',parents=[],origin='authored',
                    construction={},sources=[],labels={},reviews=[],release={'state':'draft'})
        validate_record_manifest(record,{})
        with self.assertRaises(ValueError):validate_record_manifest(record,{},for_training=True)
        record['parents']=['parent']
        with self.assertRaises(ValueError):validate_record_manifest(record,{'parent':{'family':'f','split':'locked'}})

    def test_forged_clean_reference_bytes_cannot_bypass_builder(self):
        from dataclasses import replace
        from qwen_one_two.training.inputs import canonical
        field=self.field('q','Original permitted question')
        p=self.provenance();ref=reference_input('question',{'proposition':field},p)
        forged=replace(ref,content=canonical({'role':'question','fields':{'proposition':'Smuggled current answer'}}))
        with self.assertRaises(ValueError):CleanView((forged,),(),frozenset({'q'})).materialize(p)
