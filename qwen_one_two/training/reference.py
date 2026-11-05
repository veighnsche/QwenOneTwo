"""Independent small reference weights sharing the frozen W, never a second W."""
import math
import torch
from torch import nn
from torch.func import functional_call
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
        check(not self.backbone.checkpointing, 'reference functional substitution requires outer checkpointing off')
        if mode == 'base':
            # Annotation encoding is frozen W with zero adapter contribution.
            mapping = {name: torch.zeros_like(p) for name, p in zip(self.names, self.T)}
        else:
            mapping = dict(zip(self.names, self.T))
        # functional_call scopes substitutions; no .data swaps or graph-time
        # trainability switches. The live parameters are restored on return.
        return functional_call(self.backbone, mapping, (state, 'R', embeds), strict=False)

    def annotation_states(self, ids):
        check(len(ids) <= 256, 'reference annotation length')
        with torch.no_grad():
            states, _ = self(self.backbone.initial(), 'base', self.backbone.embedding(ids))
        return states[0]

    def reconstruct(self, prefix, targets):
        """Role/slot/presence wire prefix only, followed by preceding targets.

        No source annotation prefix. Targets include the annotated end token.
        Frozen T still passes derivatives to live predicted latent inputs.
        """
        cuda_only(prefix)
        check(prefix.ndim == 2 and prefix.shape[1] == 1024 and len(prefix) > 0, 'latent prefix shape')
        check(bool(targets) and targets[-1] == 248046 and len(prefix)+len(targets) <= 512, 'reconstruction length/end')
        state = self.backbone.initial()
        hidden, state = self(state, 'T', prefix)
        losses = []
        for token in targets:
            scores, indices = self.backbone.ordinary_logits(hidden[0, -1], 1)
            target = (indices == token).nonzero().flatten()
            check(len(target) == 1, 'masked reconstruction target')
            losses.append(torch.nn.functional.cross_entropy(scores[None], target))
            hidden, state = self(state, 'T', self.backbone.embedding([token])[0])
        return torch.stack(losses).mean()

    def freeze(self, artifact_id):
        check(isinstance(artifact_id, str) and len(artifact_id) == 64, 'reference artifact hash')
        self.requires_grad_(False).eval()
        self.frozen = True
        self.artifact_id = artifact_id

    def initialize_live(self, live_codec):
        check(self.frozen, 'reference must be frozen before live initialization')
        live_codec.load_state_dict(self.codec.state_dict())
        self.assert_isolated(live_codec)
