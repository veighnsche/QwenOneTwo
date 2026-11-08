"""Direct native contracts, training-target preflight and conservative admission.

No annotation encoding in a live Session. Alignment targets are not an oracle
for the meaning of arbitrary writer vectors or the validity of a continuation.
"""
from dataclasses import dataclass
from .inputs import canonical,check,digest
from ..codec.layout import DIRECT,packet_costs,admit
from ..codec.schema import Port

GPU_CHECKS={
    'RD06-GPU-NATIVE-RETURN':'test_actual_native_return_and_prefill_source',
    'RD06-GPU-CONNECTED-GRADIENT':'test_native_loss_through_actual_return_receiver_and_writer',
}


def target_layout(references,operation,source_tokens,tokenizer):
    from .binding import UnsupportedNativeTargets
    from .direct_roles import plan_role
    try:
        planned=tuple(plan_role(tokenizer,ref) for ref in references)
        lengths={}
        for p in planned:lengths.setdefault(p['role'],[]).append(len(p['selected']))
        check(type(source_tokens)is int and source_tokens>0,'complete native prefill source token count required')
        rubric=tuple(lengths.get(role,[0])[0] for role in ('dimension','conditions','zero','one','midpoint')) if operation=='Score' else ()
        admit(packet_costs({'Noul':Port.NOUL,'Choice':Port.CHOICE,'Score':Port.SCORE}[operation],
            lengths['question'][0],lengths['target'][0],rubric=rubric,
            candidates=tuple((n,0) for n in lengths.get('alternative',[])),
            evidence=tuple(1+min(32,source_tokens-i) for i in range(0,source_tokens,32))))
        return planned
    except ValueError as exc:raise UnsupportedNativeTargets(str(exc)) from exc


def length_losses(codec,feature,roles,lengths,book,label):
    """Discrete supervised logits only; argmax lengths have no derivative."""
    from .losses import ce
    for index,(role,n) in enumerate(zip(roles,lengths)):
        book.add('structure',label+'-length-'+str(index),ce(codec.semantic_length_logits(feature,role),n))


@dataclass(frozen=True)
class NativeEpisode:
    runtime:object
    source:object
    source_digest:str
    category:str
    formation_targets:bool

    def receipt(self):
        return {'protocol':'direct-native-episode-v1','input_regime':'actual_native_prefill',
            'source':self.source.identity,'source_sha256':self.source_digest,'category':self.category,
            'formation_targets':self.formation_targets,'semantic_binding':False,
            'reference_qualification':None,'receiver_behavior_accepted':False,
            'decision_and_post_call_continuation':'masked; arbitrary native writer semantics unverified'}

    def validate(self,session,losses):
        from .binding import source_hash
        from ..codec.schema import Kind
        rt=session.runtime
        check(session.episode_report.get('release_purpose')=='training' and not session.episode_report.get('evaluation'),
              'development native evaluation cannot enter training accumulation')
        check(rt is self.runtime and rt.profile==DIRECT,'foreign native runtime/profile')
        check(source_hash(rt.registry.get(self.source,Kind.SOURCE))==self.source_digest,'native prefill source changed')
        check(session.episode_report.get('native_admission')==self.receipt(),'native receipt telemetry drift')
        allowed={'gate','writer','structure','pointer','effort'}
        if self.formation_targets:allowed.add('alignment')
        if self.category in ('direct','language_replay') and not rt.calls:allowed.add('native')
        check(set(losses)<=allowed,'unsupported native decision/Reason/continuation supervision')


def admit_native_episode(session,source,category,formation_targets):
    from .binding import source_hash
    from ..codec.schema import Kind
    check(category in ('writer','interleaved','direct','language_replay'),'native supervision category')
    return NativeEpisode(session.runtime,source,source_hash(session.runtime.registry.get(source,Kind.SOURCE)),
                         category,bool(formation_targets))
