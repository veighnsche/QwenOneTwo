"""F3 fresh permitted-view source/reference construction, offline only."""
from dataclasses import dataclass
import json
from .inputs import Binding, ReferenceInput, SLOTS, check, digest, canonical, REQUEST_KINDS, Field, reference_input


@dataclass(frozen=True)
class CleanView:
    references: tuple[ReferenceInput, ...]
    evidence: tuple[Binding, ...]
    permitted_nodes: frozenset[str]
    purpose: str = 'receiver'
    operation: str | None = None

    def materialize(self, provenance):
        check(self.purpose in {'receiver', 'reader', 'writer'}, 'clean view purpose')
        allowed = REQUEST_KINDS | ({'reason', 'approved_return'} if self.purpose == 'reader' else set())
        for ref in self.references:
            data = json.loads(ref.content)
            check(set(data) == {'role', 'fields'} and data['role'] == ref.role, 'reference serialization')
            bindings = dict(ref.bindings)
            check(len(bindings) == len(ref.bindings) and set(bindings) == set(data['fields']), 'reference field bindings')
            rebuilt = reference_input(ref.role, {name: Field(text, bindings[name]) for name,text in data['fields'].items()},
                                      provenance, permitted_nodes=self.permitted_nodes, operation=self.operation)
            check(rebuilt == ref, 'reference input/provenance drift')
            provenance.permit(ref.dependencies, allowed, self.permitted_nodes)
        excerpts = []
        for binding in self.evidence:
            keys = [key for key, node in provenance.nodes.items() if node.binding == binding]
            check(bool(keys), 'untracked clean excerpt')
            # A binding is permitted only by an explicitly selected lineage node.
            selected = [key for key in keys if key in self.permitted_nodes]
            check(bool(selected), 'excerpt outside view')
            for key in selected:
                provenance.permit((key,), allowed, self.permitted_nodes)
            excerpts.append(binding.extract(provenance.sources))
        # Global source IDs, parent offsets, hidden scenario IDs and labels are
        # deliberately absent. Local references are derived in this order.
        model_bytes = canonical({'references': [r.content.decode() for r in self.references],
                                 'sources': [x.decode() for x in excerpts], 'purpose': self.purpose, 'operation': self.operation})
        return tuple(excerpts), model_bytes


def tokenize_excerpt(tokenizer, excerpt):
    """Retokenize a derived source; reject any boundary expansion/cropping."""
    body = excerpt.decode('utf-8')
    # Render a marker separately to locate the user-content boundary reliably,
    # even if the actual excerpt also occurs in template text.
    marker = '__QOT_CLEAN_SOURCE_BOUNDARY_7a0e__'
    template = tokenizer.apply_chat_template([{'role': 'user', 'content': marker}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    check(template.count(marker) == 1, 'ambiguous template content boundary')
    prefix, suffix = template.split(marker)
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': body}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    check(rendered == prefix + body + suffix, 'template transforms source bytes')
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prefix), len(prefix) + len(body)
    selected, offsets = [], []
    for index, (a, b) in enumerate(encoded['offset_mapping']):
        if a < end and b > start:
            check(start <= a < b <= end, 'token overlaps source/template boundary')
            selected.append(index)
            offsets.append((len(body[:a-start].encode()), len(body[:b-start].encode())))
    check(bool(selected) and selected == list(range(selected[0], selected[-1]+1)), 'source token continuity')
    check(offsets[0][0] == 0 and offsets[-1][1] == len(excerpt), 'unmapped source bytes')
    # Character offsets can overlap for split UTF-8 byte tokens. Such mappings
    # cannot be passed off as verified codec byte coordinates. Reject before
    # encoding instead of inventing boundaries or expanding/cropping the view.
    check(all(0 <= a < b <= len(excerpt) for a, b in offsets), 'source mapping')
    check(all(offsets[i-1][1] == offsets[i][0] for i in range(1,len(offsets))), 'noncontiguous/overlapping source byte mapping')
    return encoded['input_ids'], selected, tuple(offsets)


class CleanEncoder:
    """No incoming cache/registry API: each encoding begins at position zero.

    Uses evaluated W+D and a frozen reference pooler. Fresh component sessions
    are constructed separately; returned registries contain only derived sources.
    No fitting, detached-contamination repair, or locked-case updates occur.
    """
    def __init__(self, backbone, reference, *, checkpoint_hash, profile_hash):
        from ..codec.neural import cuda_only
        cuda_only()
        check(len(checkpoint_hash) == len(profile_hash) == 64, 'encoder artifact identities')
        check(reference.frozen and all(not p.requires_grad for p in reference.parameters()), 'reference must be frozen')
        check(reference.artifact_id == reference.fingerprint(), 'frozen reference state changed')
        self.backbone, self.reference = backbone, reference
        self.identity = {'checkpoint': checkpoint_hash, 'profile': profile_hash,
                         'source_mode': 'D', 'reference': reference.artifact_id, 'fresh_state': True, 'kind': 'clean_component'}

    def _states(self, ids, limit):
        import torch
        check(len(ids) <= limit, 'clean encoding length exceeded; never truncate')
        state = self.backbone.initial()
        check(state.position == 0, 'nonfresh source state')
        with torch.no_grad():
            hidden, final = self.backbone(state, 'D', self.backbone.embedding(ids))
        check(final.position == len(ids) and final.lineage == state.lineage, 'clean encoding continuity')
        # State is deliberately not returned or reused in component execution.
        return hidden[0]

    def encode(self, view, provenance, *, run):
        import torch
        from ..codec.records import Registry, Source
        from ..codec.schema import Kind
        excerpts, model_bytes = view.materialize(provenance)
        registry = Registry(run)
        refs, slots, audit = [], [], []
        for index, excerpt in enumerate(excerpts):
            ids, selected, offsets = tokenize_excerpt(self.backbone.tokenizer, excerpt)
            states = self._states(ids, 512)
            body_states = states[selected]
            ref = registry.publish(Kind.SOURCE, Source(excerpt, tuple(ids[i] for i in selected), offsets, body_states))
            refs.append(ref)
            audit.append(dict(self.identity, local_source=index, input_hash=digest(excerpt),
                              tokens=len(ids), offsets=offsets, transformations=['fresh_tokenize', 'W+D', 'body_states'],
                              tensor_hash=digest(body_states.contiguous().cpu().numpy().tobytes())))
        for ref in view.references:
            text = ref.content.decode()
            rendered = self.backbone.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            ids = self.backbone.tokenizer.encode(rendered, add_special_tokens=False)
            states = self.reference.annotation_states(ids)
            with torch.no_grad():
                payload = self.reference.pooler(states, *SLOTS[ref.role])
            slots.append(payload)
            audit.append(dict(self.identity, role=ref.role, input_hash=ref.sha256,
                              transformations=['role_whitelist', 'fresh_tokenize', 'frozen_W', 'frozen_pooler'],
                              tensor_hash=digest(payload.contiguous().cpu().numpy().tobytes())))
        return {'registry': registry, 'sources': tuple(refs), 'slots': tuple(slots),
                'input_hash': digest(model_bytes), 'audit': audit}
