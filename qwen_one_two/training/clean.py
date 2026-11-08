"""F3 fresh permitted-view source/reference construction, offline only."""
from dataclasses import dataclass
import json
from .inputs import Binding, ReferenceInput, SLOTS, check, digest, canonical, REQUEST_KINDS, Field, reference_input
from .host_metrics import readback_bytes
from .frozen_source import D_SOURCE, FROZEN_SOURCE, source_mode, bind_frozen_source, source_key


@dataclass(frozen=True)
class CleanView:
    references: tuple[ReferenceInput, ...]
    evidence: tuple[Binding, ...]
    permitted_nodes: frozenset[str]
    purpose: str = 'receiver'
    operation: str | None = None
    rubric_direction: int | None = None

    def validate_request_direction(self, has_rubric, direction):
        """The renderer must use the metadata committed by this input view."""
        check((type(self.rubric_direction) is int and self.rubric_direction in (1, 2)
               and type(direction) is int and direction == self.rubric_direction) if has_rubric else self.rubric_direction is None,
              'request rubric direction differs from hashed view')

    def materialize(self, provenance):
        check(self.purpose in {'receiver', 'reader', 'writer'}, 'clean view purpose')
        check(self.rubric_direction is None or (self.operation == 'Score'
              and type(self.rubric_direction) is int and self.rubric_direction in (1, 2)),
              'clean rubric direction metadata')
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
        model_bytes = canonical({'schema': 'clean-view-v2', 'rubric_direction': self.rubric_direction,
                                 'references': [r.content.decode() for r in self.references],
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

    Defaults to evaluated W+D; opt-in frozen W retains the pooled reference. Fresh component sessions
    are constructed separately; returned registries contain only derived sources.
    No fitting, detached-contamination repair, or locked-case updates occur.
    """
    def __init__(self, backbone, reference, *, checkpoint_hash, profile_hash, cache_references=False,
                 source_policy=D_SOURCE, cache_sources=False,role_policy='frozen-reference-pooled-v1',wire_profile=None):
        from ..codec.layout import is_direct
        check(role_policy in ('frozen-reference-pooled-v1','direct-role-states-v1','direct-role-states-v2'),'unknown role policy')
        check(not is_direct(role_policy) or source_policy==FROZEN_SOURCE,'direct roles require frozen W sources')
        mode=source_mode(source_policy,cache=cache_sources)
        from ..codec.neural import cuda_only
        cuda_only()
        check(len(checkpoint_hash) == len(profile_hash) == 64, 'encoder artifact identities')
        check(reference.frozen and all(not p.requires_grad for p in reference.parameters()), 'reference must be frozen')
        check(reference.artifact_id == reference.fingerprint(), 'frozen reference state changed')
        self.backbone, self.reference = backbone, reference
        self.source_policy=source_policy
        self.role_policy=role_policy
        self.wire_profile=wire_profile
        self.source_cache={} if cache_sources else None
        self.source_cache_hits=0
        self.source_binding=bind_frozen_source(backbone) if source_policy==FROZEN_SOURCE else None
        self.reference_stamp=tuple((id(p),p._version) for p in reference.parameters())
        self.reference_cache={} if cache_references else None
        self.reference_cache_hits=0
        self.identity = {'checkpoint': checkpoint_hash, 'profile': profile_hash,
                         'source_mode': 'D', 'reference': reference.artifact_id, 'fresh_state': True, 'kind': 'clean_component'}
        if self.source_binding is not None:
            self.identity.update(source_mode=mode,source_policy=source_policy,
                frozen_source=digest(canonical(self.source_binding['identity'])))

    def _validate_source(self):
        if self.source_binding is not None:
            check(bind_frozen_source(self.backbone)['identity']==self.source_binding['identity'], 'frozen source identity changed')
            stamp=tuple((id(p),p._version) for p in self.reference.parameters())
            if stamp!=self.reference_stamp:
                # A transactional restore copies identical frozen tensors and
                # advances their version counters. Rebind only after verifying
                # the complete originally bound artifact, never a new reference.
                check(self.reference.frozen and all(not p.requires_grad for p in self.reference.parameters())
                      and self.reference.artifact_id==self.identity['reference']
                      and self.reference.fingerprint()==self.identity['reference'],
                      'frozen reference contents/ownership changed')
                self.reference_stamp=stamp

    def _states(self, ids, limit):
        import torch
        check(len(ids) <= limit, 'clean encoding length exceeded; never truncate')
        state = self.backbone.initial()
        check(state.position == 0, 'nonfresh source state')
        with torch.no_grad():
            if self.source_policy==FROZEN_SOURCE:
                self._validate_source()
                hidden, final = self.backbone.frozen_source(state,self.backbone.embedding(ids))
            else:
                hidden, final = self.backbone(state, 'D', self.backbone.embedding(ids))
        check(final.position == len(ids) and final.lineage == state.lineage, 'clean encoding continuity')
        # State is deliberately not returned or reused in component execution.
        return hidden[0]

    def _body_states(self, excerpt, ids, selected, offsets):
        import torch
        key=None
        if self.source_binding is not None:
            key=source_key(self.source_binding['identity'],excerpt,ids,selected,offsets)
        if self.source_cache is not None and key in self.source_cache:
            item=self.source_cache[key];states=item['states']
            check(states.is_cuda and states.dtype==torch.float32 and not states.requires_grad
                  and tuple(states.shape)==(len(selected),1024), 'invalid frozen source cache tensor')
            check(digest(readback_bytes(states.contiguous()))==item['tensor_hash'], 'frozen source cache mutation')
            self.source_cache_hits+=1
            return states.clone(),key
        states=self._states(ids,512)[selected]
        if self.source_cache is not None:
            frozen=states.detach().clone()
            self.source_cache[key]={'states':frozen,'tensor_hash':digest(readback_bytes(frozen.contiguous()))}
        return states,key

    def encode(self, view, provenance, *, run):
        import torch
        from ..codec.records import Registry, Source
        from ..codec.schema import Kind
        self._validate_source()
        excerpts, model_bytes = view.materialize(provenance)
        planned=None
        if self.role_policy in ('direct-role-states-v1','direct-role-states-v2'):
            from .direct_roles import preflight
            # All role/source lengths and worst-case returns are checked before
            # the first offline model forward, never after a partial encoding.
            planned=preflight(self.backbone.tokenizer,view,provenance,profile=self.role_policy)
        registry = Registry(run)
        refs, slots, audit = [], [], []
        for index, excerpt in enumerate(excerpts):
            ids, selected, offsets = tokenize_excerpt(self.backbone.tokenizer, excerpt)
            body_states,key = self._body_states(excerpt,ids,selected,offsets)
            ref = registry.publish(Kind.SOURCE, Source(excerpt, tuple(ids[i] for i in selected), offsets, body_states))
            refs.append(ref)
            binding = view.evidence[index]
            parent = provenance.sources[binding.source]
            audit.append(dict(self.identity, local_source=index, input_hash=digest(excerpt),
                              parent={'source':binding.source,'version':binding.version,'sha256':parent.sha256,
                                      'byte_start':binding.start,'byte_end':binding.end},
                              tokens=len(ids), offsets=offsets, transformations=['fresh_tokenize', 'W' if self.source_policy==FROZEN_SOURCE else 'W+D', 'body_states'],
                              tensor_hash=digest(readback_bytes(body_states.contiguous()))))
            if key is not None:audit[-1]['source_cache_key']=key
        for index,ref in enumerate(view.references):
            if planned is not None:
                item=planned['roles'][index]
                payload,key=self._body_states(ref.content,item['ids'],item['selected'],item['offsets'])
            else:payload=self.reference_payload(ref)
            slots.append(payload)
            audit.append(dict(self.identity, role=ref.role, input_hash=ref.sha256, bindings=ref.bindings, dependencies=ref.dependencies,
                              transformations=['role_whitelist', 'fresh_tokenize', 'frozen_W', 'frozen_pooler'],
                              tensor_hash=digest(readback_bytes(payload.contiguous()))))
            if planned is not None:
                audit[-1].update(transformations=['role_whitelist','fresh_tokenize','frozen_W','body_states'],
                    role_profile=self.role_policy,source_cache_key=key,selected=tuple(item['selected']),offsets=item['offsets'])
        result={'registry': registry, 'sources': tuple(refs), 'slots': tuple(slots),
                'input_hash': digest(model_bytes), 'audit': audit}
        if planned is not None:result.update(role_profile=self.role_policy,admission=planned['costs'])
        if self.wire_profile is not None:result['role_profile']=self.wire_profile
        return result

    def reference_payload(self,ref):
        """Cache only frozen W/reference outputs; D-dependent sources stay fresh."""
        import torch
        check(self.role_policy not in ('direct-role-states-v1','direct-role-states-v2') or ref.role=='reason',
              'direct request roles cannot use pooled reference payloads')
        check(self.reference.frozen and self.reference.artifact_id==self.identity['reference']
              and all(not p.requires_grad for p in self.reference.parameters()),'reference cache identity/freeze')
        self._validate_source()
        key=ref.sha256 if self.source_binding is None else digest(canonical({
            'frozen':self.source_binding['identity'],'reference':self.reference.artifact_id,
            'role':ref.role,'input':ref.sha256,'slots':SLOTS[ref.role]}))
        if self.reference_cache is not None and key in self.reference_cache:
            self.reference_cache_hits+=1
            return self.reference_cache[key].clone()
        rendered=self.backbone.tokenizer.apply_chat_template([{'role':'user','content':ref.content.decode()}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=self.backbone.tokenizer.encode(rendered,add_special_tokens=False)
        states=self.reference.annotation_states(ids)
        with torch.no_grad():payload=self.reference.pooler(states,*SLOTS[ref.role])
        if self.reference_cache is not None:self.reference_cache[key]=payload.detach().clone()
        return payload
