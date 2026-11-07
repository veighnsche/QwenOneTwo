"""Execute reviewed native episodes without a textual decision-model prompt.

No optimizer, generation service or launch-on-import. Offline request annotations
select supervised *control actions* only. The writer emits every semantic slot
from live R states; Qev receives codec packets and its actual return is consumed.
Unverified request meanings do not acquire outcome labels by matching an ID.
"""
from dataclasses import dataclass, replace
import uuid

from .inputs import Exposure, canonical, check, digest
from .profile import PHASES, guidance_prefix
from .release import ReviewedRelease


class PrerequisiteError(ValueError):
    pass


@dataclass(frozen=True)
class NativePlan:
    record: str
    category: str
    prompt: str
    calls: tuple
    during_answer: bool
    target: str | None
    effort: float | None
    override: int | None
    limit: int
    release: str


def native_plan(release, row, phase, window):
    """Select permitted inputs before any model execution; never serialize row."""
    check(isinstance(release, ReviewedRelease), 'load a reviewed release first')
    check(canonical(row) == canonical(release.record(row['id'])), 'record/release drift')
    check(row['id'] in release.selected_ids,
          'unselected episode record')
    check(release.purpose != 'locked', 'locked collection is not an episode-development source')
    check(phase in PHASES and phase != 'reference' and 1 <= window <= PHASES[phase], 'native phase/window')
    category = row['training_category']
    if category in {'reference', 'receiver', 'reader', 'decision_replay'}:
        raise PrerequisiteError('use the role-specific reference/component producer for '+category)
    check(category in {'writer', 'interleaved', 'direct', 'language_replay'}, 'native category')
    fields = row['model_inputs']
    allowed = {'causal_history', 'evidence', 'draft', 'counters'} if category in {'writer', 'interleaved'} else {'prompt', 'evidence'}
    check(set(fields) <= allowed, 'native input whitelist')
    prompt = fields['causal_history'] if category in {'writer', 'interleaved'} else fields['prompt']
    check(isinstance(prompt,str) and prompt.strip(), 'native human prompt')
    check(prompt == row['human_input']['prompt'], 'human prompt/input drift')
    # Evidence is supplied task data. No request_target/result/Reason/gloss is
    # appended. Multi-case prompts already contain the full scoped evidence.
    evidence = fields.get('evidence', {})
    if set(evidence) <= {'facts', 'rules'}:
        for kind in ('facts', 'rules'):
            values = evidence.get(kind, [])
            check(isinstance(values,list) and all(isinstance(v,str) for v in values), 'native evidence')
            missing = [v for v in values if v not in prompt]
            if missing:
                prompt += '\n'+kind.title()+':\n'+'\n'.join(missing)
    else:
        check(category == 'interleaved' and all(isinstance(v,dict) for v in evidence.values()), 'scoped evidence')
        check(all(text in prompt for part in evidence.values() for values in part.values() for text in values),
              'scoped evidence absent from human prompt')
    calls = row.get('trajectory', {}).get('calls', []) if category == 'interleaved' else []
    if category == 'writer':
        calls = [{'operation': row['operation'], 'request_target':row['supervision']['request_semantics']}]
    # Carry only structure, never the English annotations, into execution.
    controls = []
    for call in calls:
        semantics = call['request_target']
        operation = call['operation']
        check(operation in {'Noul','Choice','Score'}, 'operation target')
        count = len(semantics.get('alternatives', []))
        check((1 <= count <= 16) if operation == 'Choice' else count == 0, 'candidate count target')
        direction = semantics.get('rubric', {}).get('direction', 'higher')
        check(direction in {'higher','lower'}, 'rubric direction')
        controls.append((operation, count, direction))
    if phase == 'components':
        controls = controls if category == 'writer' else []
    else:
        controls = list(guidance_prefix(phase,window,controls))
    # Post-call prose depends on the actual request and return. No automatic
    # transfer of a source-case answer to arbitrary predicted meanings.
    target = None
    if category in {'direct','language_replay'}:
        target = row['supervision']['continuations'][0]
    effort_review = row.get('initial_effort_review', {})
    effort = effort_review.get('preference') if category in {'direct','interleaved'} and effort_review.get('eligible') is True else None
    return NativePlan(row['id'],category,prompt,tuple(controls),
                      row.get('trajectory',{}).get('invocation_window') == 'during_answer',
                      target,effort,512 if phase in {'components','guided_joint'} else None,
                      512 if phase == 'components' else 1024,release.identity)


class EpisodeProducer:
    """Driver callback for native writer/direct/replay/joint episodes.

    Components requiring a frozen semantic reference reject if it is missing.
    This is intentionally not an implicit claim that all A4 stages are ready.
    Training guidance changes actions/structure, never substitutes latent slots
    or the evaluator result. Free phases run the production gate and writer.
    """
    def __init__(self, backbone, codec, release, *, reference=None, checkpoint_hash=None, profile_hash=None,
                 reference_evidence=None,diagnostic_reference=False):
        check(isinstance(release,ReviewedRelease), 'load a reviewed release first')
        self.backbone,self.codec,self.release = backbone,codec,release
        self.reference,self.reference_evidence=reference,reference_evidence
        self.diagnostic_reference=diagnostic_reference
        self.component = None
        from .reference_qualification import ReferenceEvidence
        check(reference_evidence is None or (isinstance(reference_evidence,ReferenceEvidence) and reference is not None),
              'load measured reference evidence for this artifact')
        check(not (reference_evidence is not None and diagnostic_reference),'qualified/diagnostic reference modes are distinct')
        if reference is not None:
            from .reference_qualification import ReferenceEvidence
            if isinstance(reference_evidence,ReferenceEvidence):
                reference_evidence.assert_reference(reference)
                reference_evidence.assert_receiver(backbone,codec)
            from .component_producer import ComponentProducer
            self.component=ComponentProducer(backbone,codec,release,reference,
                checkpoint_hash=checkpoint_hash,profile_hash=profile_hash,
                reference_evidence=reference_evidence,diagnostic_reference=diagnostic_reference)
        self.last_report = None

    def __call__(self,row,phase,window):
        if self.reference_evidence is not None:self.reference_evidence.assert_reference(self.reference)
        check(canonical(row)==canonical(self.release.record(row['id'])),'record/release drift')
        check(self.release.purpose!='locked','locked collection is not an episode-development source')
        check(row['id'] in self.release.selected_ids, 'unselected episode record')
        check(phase in PHASES and 1<=window<=PHASES[phase],'phase/window')
        if row['training_category'] in {'receiver','reader','decision_replay'}:
            if self.component is None:
                raise PrerequisiteError('isolated components require a frozen reference artifact; no reference fitting is authorized')
            result=self.component(row,phase,window)
            self.last_report=self.component.last_report
            return result
        import torch
        from ..codec.session import Session
        from ..codec.runtime import Runtime
        from ..codec.records import Source
        from ..codec.schema import Kind
        from .clean import tokenize_excerpt
        from .losses import ExampleLosses, regression

        plan = native_plan(self.release,row,phase,window)
        ids,selected,offsets = tokenize_excerpt(self.backbone.tokenizer,plan.prompt.encode())
        check(len(ids)+6 < plan.limit, 'prompt exceeds complete-episode ceiling; never truncate')
        class BoundedRuntime(Runtime):
            def debit(self,kind,count,label):
                # Guard before ledger mutation, not after charging a forward
                # that will never execute. The production context is unchanged.
                if self.position+count>plan.limit:
                    raise EpisodeLengthError('complete episode exceeds training ceiling')
                return super().debit(kind,count,label)
        rt = BoundedRuntime(str(uuid.uuid4()),len(ids),0,128)
        session = Session(self.backbone,self.codec,rt)
        h = session.prefill(ids)
        source = rt.registry.publish(Kind.SOURCE,Source(plan.prompt.encode(),tuple(ids[i] for i in selected),
                                                       offsets,h[0,selected]))
        session.start_budget(plan.override)
        book = ExampleLosses()
        if plan.effort is not None:
            book.add('effort','initial',regression(session.raw_budget,point=plan.effort))
        report = {'producer':'native-episodes-v1','record':plan.record,'release_sha256':plan.release,
                  'prompt_sha256':digest(plan.prompt.encode()),'source_offsets':offsets,
                  'source_token_ids':tuple(ids[i] for i in selected),'calls':[],
                  'semantic_losses':'masked: actual writer meanings have no verified semantic binding',
                  'continuation_loss':'masked after predicted requests/returns',
                  'correction_losses':'masked: prescribed interventions are not observed corrections'}
        session.episode_report = report
        report['diagnostic_reference']=self.diagnostic_reference
        qualified=self.reference_evidence is not None and not self.diagnostic_reference
        report['receiver_behavior_accepted']=qualified
        self.last_report = report
        teacher_forcing=plan.target is not None and phase in {'components','guided_joint'}
        compatible_prefix=bool(plan.calls)
        try:
            if plan.during_answer and plan.calls:
                session.close()
            for index,control in enumerate(plan.calls):
                targets=None
                if self.reference is not None:
                    from .binding import request_targets
                    targets=request_targets(self.reference,self.release,row,plan.prompt.encode(),call_index=index,
                        source_ref=source,source_record=rt.registry.get(source,Kind.SOURCE))
                # One incompatible return ends the compatible guidance prefix.
                # Without semantic verification no later annotated call is safe.
                if not self._guided_call(session,source,control,book,report,plan.limit,targets=targets):
                    compatible_prefix=False
                    break
                if not report['calls'][-1].get('compatible_return',False):
                    compatible_prefix=False
                    report['guidance_stopped']='first return has no verified continuation binding'
                    break
            expected=len(row.get('trajectory',{}).get('calls',[])) if plan.category=='interleaved' else 1
            if compatible_prefix and len(plan.calls)==expected and phase in {'components','guided_joint'}:
                plan=replace(plan,target=row['supervision']['continuations'][0])
                teacher_forcing=True
                report['continuation_loss']='supported by exact request and actual return bindings'
            # Direct and language replay teacher forcing are safe only while
            # no model decision has changed the task state. Afterwards use the
            # real gate/native path, with unsupported continuation losses masked.
            if teacher_forcing and (not rt.calls or compatible_prefix):
                if rt.phase==0:session.close()
                targets = self.backbone.tokenizer.encode(plan.target,add_special_tokens=False)
                check(len(targets) <= rt.answer, 'answer target exceeds A; never truncate')
                if rt.position+len(targets)+1 > plan.limit:
                    self._overlength(session,report)
                else:
                    losses,_ = session.native(targets,True)
                    for index,loss in enumerate(losses):book.add('native',str(index),loss)
                    # Predicted termination is supervised; mechanical closure is not.
                    scores,indices=self.backbone.ordinary_logits(session.hidden[0,-1],rt.phase)
                    from .losses import ce
                    end=(indices==248046).nonzero().flatten()
                    check(len(end)==1,'chat end must be an ordinary answer target')
                    book.add('native','end',ce(scores,end[0]))
                    rt.debit('H',1,'chat_end')
                    session.step(self.backbone.embedding([248046])[0],'R','chat_end',False)
                    rt.terminate('Completed',emit_end=False)
            else:
                while rt.terminal is None:
                    if rt.position >= plan.limit:
                        self._overlength(session,report)
                        break
                    session.native_step()
        except EpisodeLengthError:
            self._overlength(session,report)
        except ValueError as exc:
            rt.terminate('Failed',usable=False,emit_end=False)
            report['failure']=str(exc)
        complete=rt.terminal in {'Completed','Limited'}
        losses=book.finish() if complete else {}
        exposure=Exposure('joint_receiver' if plan.category=='interleaved' else 'native_writer',
                          native_teacher_forcing=teacher_forcing,
                          decision_forcing=bool(plan.calls) or teacher_forcing,budget_override=plan.override)
        exposure.validate()
        report.update(summary=session.summary(),eligible_components=sorted(losses),
                      actual_returns_only=True)
        return session,losses,exposure

    @staticmethod
    def _overlength(session,report):
        report['failure']='complete episode exceeds training ceiling; all prefix losses discarded'
        session.runtime.terminate('Failed',usable=False,emit_end=False)

    def _guided_call(self,s,source,control,book,report,limit,targets=None):
        """Teacher-forced structure with live R semantic emissions, actual D result."""
        import torch
        from ..codec.schema import Port,Kind,Role
        from ..codec.records import Semantic,Span
        from ..codec.writer import decode_result
        from .losses import ce
        rt=s.runtime
        operation,count,direction=control
        port={'Noul':Port.NOUL,'Choice':Port.CHOICE,'Score':Port.SCORE}[operation]
        if rt.b==rt.budget or rt.attempts>=16:return False
        if rt.position+2>limit:raise EpisodeLengthError()
        f=self.codec.feature(s.hidden[0,-1],rt.counters())
        book.add('gate','intent',ce(self.codec.gate(f),int(port)))
        writer=s.begin_writer(port)
        f=writer.feature(s.hidden[0,-1])
        # Explicit construct/no-attribute targets, not arbitrary untrained
        # Decoder bytes. Every semantic meaning is still a connected live emit.
        book.add('structure','record_mode',ce(self.codec.heads['record_mode'](f),0))
        book.add('structure','child_count',ce(self.codec.heads['child_count'](f),0))
        q=rt.publish(Kind.QUESTION,Semantic(self.codec.emit(f,Role.QUESTION,8)))
        t=rt.publish(Kind.TARGET,Semantic(self.codec.emit(f,Role.TARGET,2)))
        writer.draft=replace(writer.draft,question=q,target=t)
        def tick(action):
            if rt.position+1>limit:raise EpisodeLengthError()
            s.step(writer.tick_vector(s.hidden[0,-1])[None],'R','writer_tick',False)
            f=writer.feature(s.hidden[0,-1])
            book.add('writer',str(rt.call.writer),ce(self.codec.writer_action(f),action))
            return f
        # Source coordinates are actual prefill token offsets, not characters
        # mislabelled as tokens. Split at the transport's 32-token span bound.
        n=len(rt.registry.get(source).token_ids)
        # More than eight spans cannot describe the full source in this profile.
        # Fail rather than silently discard some evidence or invent an excerpt.
        if n>256:
            raise ValueError('source exceeds eight complete evidence spans; no cropped training exposure')
        for start in range(0,n,32):
            tick(0)
            writer.draft=replace(writer.draft,evidence=writer.draft.evidence+(Span(source,start,min(32,n-start)),))
            rt.writer_action(0)
        for index in range(count):
            f=tick(1)
            ref=rt.publish(Kind.ALTERNATIVE,Semantic(self.codec.emit(f,Role.CANDIDATE,4)))
            writer.draft=replace(writer.draft,candidates=writer.draft.candidates+(ref,))
            rt.writer_action(1)
        if port==Port.SCORE:
            f=tick(2)
            value=1 if direction=='higher' else 2
            book.add('structure','direction',ce(self.codec.heads['direction'](f),value))
            slots=torch.cat([self.codec.emit(f,r,2) for r in (Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE)])
            slots=torch.cat((slots,slots.new_zeros(2,1024)))
            ref=rt.publish(Kind.RUBRIC,Semantic(slots,direction=value,midpoint=False))
            writer.draft=replace(writer.draft,rubric=ref);rt.writer_action(2)
        tick(5)
        d=writer.draft
        packet=rt.seal(d.question,d.target,d.evidence,d.rubric,d.candidates)
        if rt.position+rt.call.position_hold>limit:raise EpisodeLengthError()
        labels=None;binding={'supported':False,'reason':'reviewed reference targets unavailable'}
        if targets is not None:
            from .binding import formation_losses,bind_labels
            formation_losses(self.reference,rt.registry,packet.snapshot,targets,book)
            labels,binding=bind_labels(self.reference,self.reference_evidence,rt.registry,packet.snapshot,targets,runtime=rt)
        inputs,evaluated,_=s.evaluate(packet)
        item={'operation':operation,'snapshot':packet.snapshot.identity,'request_positions':len(packet),
              'input_shape':list(inputs.shape),'input_dtype':str(inputs.dtype),'cuda':inputs.is_cuda,
              'text_tokens_for_invocation':0,'predicted_status':int(evaluated['status_logits'].argmax())+1,
              'selected_by':'actual Qev heads','semantic_targets_applied':labels is not None,'binding':binding}
        report['calls'].append(item)
        returned=decode_result(self.codec,rt,evaluated)
        if labels is not None:
            from .episodes import decision_losses
            from .binding import compatible_return
            actual=rt.registry.get(rt.call.result,Kind.RESULT).status
            values,masks=decision_losses(rt.registry,packet.snapshot,actual,evaluated,labels)
            for name,value in values.items():
                book.add(name,'call-'+str(rt.call.ordinal),value,role='reason' if name=='alignment' else None)
            item['eligibility']=masks
            item['compatible_return']=compatible_return(rt.registry,rt.call.result,labels,targets)
        delivered=s.deliver(returned)
        item.update(return_positions=len(returned) if returned is not None else 0,consumed=delivered)
        return delivered


class EpisodeLengthError(RuntimeError):
    """No truncation/backward of a partial training episode."""
