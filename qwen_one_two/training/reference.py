"""Independent small reference weights sharing the frozen W, never a second W."""
import math
import hashlib
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from ..codec.neural import Codec, ReferencePooler, cuda_only
from .inputs import check


class Reference(nn.Module):
    def __init__(self, backbone, live_codec):
        super().__init__()
        cuda_only()
        # Deliberately not registered: W/R/D belong to the live backbone owner.
        object.__setattr__(self, 'backbone', backbone)
        self.names = tuple(n for n, p in backbone.named_parameters() if n.endswith(('.R_A', '.R_B')))
        originals = dict(backbone.named_parameters())
        self.T = nn.ParameterList()
        for name in self.names:
            p = torch.empty_like(originals[name], dtype=torch.float32)
            if name.endswith('.R_A'):
                nn.init.uniform_(p, -1/math.sqrt(p.shape[1]), 1/math.sqrt(p.shape[1]))
            else:
                nn.init.zeros_(p)
            self.T.append(nn.Parameter(p))
        with torch.device('cuda'):
            self.pooler = ReferencePooler()
            self.codec = Codec(backbone.sigma, backbone.model.model.embed_tokens.weight,
                               {'q.weight': live_codec.qev_q.weight.detach(), 'k.weight': live_codec.qev_k.weight.detach()})
        # Reference codec starts compatible, owns no aliases to live weights.
        self.codec.load_state_dict(live_codec.state_dict())
        self.frozen = False
        self.artifact_id = None
        self.assert_isolated(live_codec)

    def assert_isolated(self, live_codec):
        live = {p.untyped_storage().data_ptr() for p in live_codec.parameters()}
        check(all(p.untyped_storage().data_ptr() not in live for p in self.parameters()), 'reference/live storage alias')
        base = {id(p) for p in self.backbone.parameters()}
        check(not any(id(p) in base for p in self.parameters()), 'reference registered backbone')

    def forward(self, state, mode, embeds):
        check(mode in {'base', 'T'}, 'reference mode')
        if mode == 'base':
            # Annotation encoding is frozen W with zero adapter contribution.
            mapping = {name: torch.zeros_like(p) for name, p in zip(self.names, self.T)}
        else:
            mapping = dict(zip(self.names, self.T))
        # functional_call scopes substitutions; no .data swaps or graph-time
        # trainability switches. The live parameters are restored on return.
        # A live layer checkpoint must never replay after the T mapping has
        # been restored to R. Wrap the whole scoped call instead, so replay
        # reinstalls precisely the same explicit T/base parameters.
        def scoped(current_state, current_embeds):
            outer = self.backbone.checkpointing
            self.backbone.checkpointing = False
            try:
                return functional_call(self.backbone, mapping, (current_state, 'R', current_embeds), strict=False)
            finally:
                self.backbone.checkpointing = outer
        if self.backbone.checkpointing and torch.is_grad_enabled():
            return checkpoint(scoped,state,embeds,use_reentrant=False)
        return scoped(state,embeds)

    def annotation_states(self, ids):
        check(len(ids) <= 256, 'reference annotation length')
        with torch.no_grad():
            states, _ = self(self.backbone.initial(), 'base', self.backbone.embedding(ids))
        return states[0]

    def reconstruct(self, prefix, targets, *, target_history=True):
        """Role/slot/presence wire prefix only, followed by preceding targets.

        No source annotation prefix. Targets include the annotated end token.
        Frozen T still passes derivatives to live predicted latent inputs.
        """
        cuda_only(prefix)
        check(prefix.ndim == 2 and prefix.shape[1] == 1024 and len(prefix) > 0, 'latent prefix shape')
        check(bool(targets) and targets[-1] == 248046 and len(prefix)+len(targets) <= 512, 'reconstruction length/end')
        state = self.backbone.initial()
        hidden, state = self(state, 'T', prefix)
        # One frozen allowed-vocabulary slice per auxiliary episode. Indexing
        # the tied table afresh at every target would retain ~485 MiB copies
        # in each projection's autograd context without changing any values.
        indices = torch.cat((torch.arange(248044,device='cuda'),torch.tensor([248046],device='cuda')))
        vocabulary = self.backbone.model.model.embed_tokens.weight[indices]
        check(not vocabulary.requires_grad, 'frozen reconstruction vocabulary')
        losses = []
        for token in targets:
            h = hidden[0,-1]
            if self.backbone.compensated and vocabulary.dtype == torch.bfloat16:
                from ..g1_precision_math import compensated_linear
                from ..codec.precision import linear
                project = linear if self.backbone.precision_parts == 3 else compensated_linear
                scores = project(h,vocabulary)
            else:
                scores = torch.nn.functional.linear(h.to(vocabulary.dtype),vocabulary).float()
            target = (indices == token).nonzero().flatten()
            check(len(target) == 1, 'masked reconstruction target')
            losses.append(torch.nn.functional.cross_entropy(scores[None], target))
            preceding=self.backbone.embedding([token])[0] if target_history else prefix.new_zeros((1,1024))
            hidden, state = self(state, 'T', preceding)
        return torch.stack(losses).mean()

    def reconstruct_teacher_forced(self,prefix,targets,*,target_history=True):
        """Same causal objective in one transformer sequence, microbatch one.

        Position n predicts target n from preceding targets only. The source
        annotation is never a decoder prefix. Vocabulary arithmetic is unchanged.
        Serial reconstruct remains available for parity and historical evidence.
        """
        cuda_only(prefix)
        check(prefix.ndim==2 and prefix.shape[1]==1024 and len(prefix)>0,'latent prefix shape')
        check(targets and targets[-1]==248046 and len(prefix)+len(targets)<=512,'reconstruction length/end')
        check(all(type(t) is int and (0<=t<248044 or t==248046) for t in targets),'masked reconstruction target')
        check(type(target_history) is bool,'target-history exposure flag')
        preceding=(self.backbone.embedding(targets[:-1])[0] if target_history else prefix.new_zeros((len(targets)-1,1024))) if len(targets)>1 else prefix.new_empty((0,1024))
        hidden,_=self(self.backbone.initial(),'T',torch.cat((prefix,preceding)))
        hidden=hidden[0,len(prefix)-1:]
        indices=torch.cat((torch.arange(248044,device='cuda'),torch.tensor([248046],device='cuda')))
        vocabulary=self.backbone.model.model.embed_tokens.weight[indices]
        check(not vocabulary.requires_grad,'frozen reconstruction vocabulary')
        if self.backbone.compensated and vocabulary.dtype==torch.bfloat16:
            from ..g1_precision_math import compensated_linear
            from ..codec.precision import linear
            scores=(linear if self.backbone.precision_parts==3 else compensated_linear)(hidden,vocabulary)
        else:scores=torch.nn.functional.linear(hidden.to(vocabulary.dtype),vocabulary).float()
        labels=torch.tensor([248044 if t==248046 else t for t in targets],device='cuda')
        return torch.nn.functional.cross_entropy(scores,labels)

    @torch.no_grad()
    def exact_recovery(self,prefix,targets):
        """Conservative read-only recovery of an existing reviewed annotation.

        Stop at the first greedy mismatch: the exact-match outcome is already
        determined. No target is ever supplied before the model predicts it.
        Nonexact paraphrases remain unresolved, never automatically approved.
        """
        cuda_only(prefix)
        check(self.frozen and self.artifact_id==self.fingerprint(),'frozen unchanged reference required')
        check(targets and targets[-1]==248046 and len(prefix)+len(targets)<=512,'recovery length/end')
        hidden,state=self(self.backbone.initial(),'T',prefix)
        indices=torch.cat((torch.arange(248044,device='cuda'),torch.tensor([248046],device='cuda')))
        vocabulary=self.backbone.model.model.embed_tokens.weight[indices]
        from ..codec.precision import linear
        from ..g1_precision_math import compensated_linear
        for index,target in enumerate(targets):
            project=linear if self.backbone.precision_parts==3 else compensated_linear
            scores=project(hidden[0,-1],vocabulary)
            check(bool(scores.isfinite().all()),'nonfinite recovery logits')
            token=int(indices[scores.argmax()])
            if token!=target:
                return {'exact':False,'matched_tokens':index,'target_tokens':len(targets),
                        'first_mismatch':{'index':index,'predicted_token':token,'target_token':target}}
            if index+1<len(targets):hidden,state=self(state,'T',self.backbone.embedding([token])[0])
        return {'exact':True,'matched_tokens':len(targets),'target_tokens':len(targets),'first_mismatch':None}

    def fingerprint(self):
        return state_fingerprint(self.state_dict())

    def freeze(self, artifact_id=None):
        actual = self.fingerprint()
        check(artifact_id is None or artifact_id == actual, 'reference state hash mismatch')
        artifact_id = actual
        self.requires_grad_(False).eval()
        self.frozen = True
        self.artifact_id = artifact_id

    def initialize_live(self, live_codec, evidence=None):
        from .reference_qualification import ReferenceEvidence
        check(isinstance(evidence,ReferenceEvidence),'measured reference qualification required before live initialization')
        evidence.assert_reference(self)
        check(self.frozen and self.artifact_id == self.fingerprint(), 'reference must be frozen and unchanged before live initialization')
        live_codec.load_state_dict(self.codec.state_dict())
        self.assert_isolated(live_codec)


def reference_objective(reference, annotation, provenance, *, operation=None, contrast_candidates=None, positives=(), negatives=()):
    """Reference preparation objective over a previously F7-validated record.

    Returns one normalized example loss per component. This builds a graph;
    it does not perform an optimizer update or assert semantic approval.
    """
    from ..codec.schema import Role
    from .inputs import SLOTS
    from .clean import CleanView
    CleanView((annotation,),(),frozenset(annotation.dependencies),
              purpose="reader" if annotation.role == "reason" else "receiver", operation=operation).materialize(provenance)
    from .losses import contrast
    check(not reference.frozen, 'cannot fit a frozen reference artifact')
    b = reference.backbone
    text = annotation.content.decode('utf-8')
    rendered = b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                              add_generation_prompt=True,enable_thinking=True)
    ids = b.tokenizer.encode(rendered,add_special_tokens=False)
    states = reference.annotation_states(ids)
    payload = reference.pooler(states,*SLOTS[annotation.role])
    roles = {'question':Role.QUESTION,'target':Role.TARGET,'alternative':Role.CANDIDATE,
             'dimension':Role.DIMENSION,'conditions':Role.CONDITIONS,'zero':Role.ZERO,
             'one':Role.ONE,'midpoint':Role.MIDPOINT,'key':Role.KEY,'reason':Role.REASON}
    role = roles[annotation.role]
    prefix = torch.stack([reference.codec.wire(vector,role,i,0,0 if annotation.role == 'key' else None)
                          for i,vector in enumerate(payload)])
    targets = b.tokenizer.encode(text,add_special_tokens=False)+[248046]
    losses = {'reconstruction':reference.reconstruct(prefix,targets)}
    if contrast_candidates is not None:
        value = contrast(payload,contrast_candidates,positives,negatives)
        if value is not None:
            losses['contrast'] = value
    return losses,payload


def state_fingerprint(state):
    """Content identity of named small tensors, independent of serialization."""
    from .inputs import canonical
    h = hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(canonical({'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype)}))
        # Host byte hashing is artifact bookkeeping, not CPU inference.
        h.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
