"""Fixed, read-only input transforms of existing bound source count literals."""
from dataclasses import replace
import json
import re
from .inputs import Binding, Node, Provenance, Field, reference_input, canonical, digest, check

PROTOCOL = 'numeric-access-comparison-v1'
ARMS = ('literal', 'int64')


def legacy_hash(body):
    """Explicit diagnostic mapping only; never an admission/cache fallback."""
    value=json.loads(body)
    check(value.pop('schema')=='clean-view-v2', 'explicit v2 migration only')
    value.pop('rubric_direction')
    check(set(value)=={'references','sources','purpose','operation'}, 'legacy view shape')
    return digest(canonical(value))


def count_specs(excerpts):
    """Exact source-template extraction; no outcome/annotation/operation routing."""
    check(len(excerpts)>=2, 'two existing count facts required')
    matches=[re.fullmatch(r'The (requirement) is ([0-9]+) (.+)\.', excerpts[0].decode()),
             re.fullmatch(r'([0-9]+) (.+) are (complete)\.', excerpts[1].decode())]
    check(all(matches), 'unsupported source count template')
    check(matches[0][3]==matches[1][2], 'count units differ')
    specs=[]
    for i,(match,groups) in enumerate(zip(matches,((1,2,3),(3,1,2)))):
        def part(group):
            a,b=match.span(group); text=excerpts[i].decode()
            return {'text':match[group], 'start':len(text[:a].encode()), 'end':len(text[:b].encode())}
        prop,value,unit=(part(group) for group in groups)
        check(1<=len(value['text'].encode())<=8 and int(value['text'])<2**63, 'one-atom literal / INT64 range')
        specs.append({'source':i,'property':prop,'literal':value,'unit':unit})
    return specs


def byte_span(offsets, start, end):
    """Exact substring coordinates in a verified clean token map; no expansion."""
    selected=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    check(selected and selected==list(range(selected[0],selected[-1]+1)), 'literal token span')
    first,last=selected[0],selected[-1]
    check(offsets[first][0]<=start<end<=offsets[last][1], 'literal outside token map')
    return first,len(selected),start-offsets[first][0],end-offsets[last][0]


class NumericConstruction:
    def __init__(self, arm):
        check(arm in ARMS, 'single declared numeric arm')
        self.arm=arm; self.receipt=None

    def prepare(self, view, provenance, indices):
        check(view.operation=='Score' and view.purpose=='receiver', 'Score receiver only')
        excerpts,body=view.materialize(provenance); specs=count_specs(excerpts)
        nodes=dict(provenance.nodes); refs=list(view.references)
        for spec in specs:
            base=view.evidence[spec['source']]; fields={}
            for name in ('property','unit'):
                part=spec[name]; key=f'numeric-{spec["source"]}-{name}'
                check(key not in nodes, 'derived source node collision')
                bound=Binding(base.source,base.version,base.start+part['start'],base.start+part['end'])
                check(bound.extract(provenance.sources)==part['text'].encode(), 'verbatim key binding')
                nodes[key]=Node('source_fact',digest(part['text'].encode()),binding=bound)
                fields[name]=Field(part['text'],key)
            p=Provenance(nodes,provenance.sources)
            ref=reference_input('key',fields,p,operation='Score')
            spec['key_index']=len(refs);spec['key_sha256']=ref.sha256;refs.append(ref)
        p=Provenance(nodes,provenance.sources)
        augmented=replace(view,references=tuple(refs),permitted_nodes=frozenset(nodes))
        _,augmented_body=augmented.materialize(p)
        self.receipt={'profile':PROTOCOL,'arm':self.arm,'source_view_sha256':digest(body),
            'augmented_view_sha256':digest(augmented_body),'attributes':specs,
            'input_sha256':digest(canonical({'view':digest(augmented_body),'profile':PROTOCOL,'arm':self.arm,'attributes':specs})),
            'new_annotation_or_label':False,'keys':'verbatim existing source property/unit substrings'}
        return augmented,p,dict(indices,numeric_specs=specs)

    def request(self, factory, indices):
        """Return a bound request builder; targets/gold values are inaccessible."""
        import torch
        from ..codec.schema import ValueKind, Role
        check(not torch.is_grad_enabled(), 'numeric comparison is read-only')
        attrs=[]
        for spec in indices['numeric_specs']:
            source=factory.registry.get(factory.sources[spec['source']])
            part=spec['literal']; args=byte_span(source.offsets,part['start'],part['end'])
            span=factory.span(spec['source'],*args)
            _,literal,_,_=factory.registry.span(span)
            check(literal==part['text'].encode(), 'literal source byte drift')
            kind=ValueKind.INT64 if self.arm=='int64' else ValueKind.UTF8
            attr=factory.source_attribute(spec['key_index'],span,kind)
            check(attr.cost==4 and attr.value.data==(int(literal) if self.arm=='int64' else literal), 'exact matched numeric value')
            attrs.append(attr)

        def build(*args,**kwargs):
            packet=factory.request(*args,**kwargs,target_attributes=tuple(attrs))
            numeric=[p for p in packet.positions if p.role in (Role.INT64,Role.UTF8)]
            check(len(numeric)==2 and sum(p.role==Role.ATTRIBUTE_DESC for p in packet.positions)==2, 'two exact numeric attributes')
            for position,spec in zip(numeric,indices['numeric_specs']):
                literal=spec['literal']['text'].encode()
                encoded=int(literal).to_bytes(8,'big',signed=True) if self.arm=='int64' else literal.ljust(8,b'\0')
                check(position.payload.data==encoded, 'canonical literal atom bytes')
            self.receipt.update(atom_hex=[p.payload.data.hex() for p in numeric],added_request_positions=8)
            return packet
        return build


def frozen_encoder(backbone, reference, capture, roles, keys, *, receiver, profile_hash):
    """Reuse captured pools; cache fresh fixed-D source states only in this run."""
    import torch
    from .clean import CleanEncoder
    from .inputs import SLOTS

    class FrozenEncoder(CleanEncoder):
        def __init__(self):
            super().__init__(backbone,reference,checkpoint_hash=receiver,profile_hash=profile_hash)
            self.source_cache={};self.key_cache={}
            self.stats={'source_encodings':0,'source_tokens':0,'source_cache_hits':0,
                        'reference_encodings':0,'reference_tokens':0,'reference_cache_hits':0}

        def _states(self, ids, limit):
            check(not torch.is_grad_enabled() and all(not p.requires_grad for p in backbone.parameters()), 'frozen source cache only')
            key=tuple(ids)
            if key not in self.source_cache:
                self.source_cache[key]=super()._states(ids,limit)
                self.stats['source_encodings']+=1;self.stats['source_tokens']+=len(ids)
            else:self.stats['source_cache_hits']+=1
            return self.source_cache[key].clone()

        def reference_payload(self, ref):
            if ref.sha256 in roles:
                check(roles[ref.sha256]==ref.role, 'saved reference role')
                self.stats['reference_cache_hits']+=1
                return capture['reference.'+ref.sha256+'.pool'].clone()
            check(ref.role=='key' and ref.sha256 in keys and not torch.is_grad_enabled(), 'only planned verbatim keys may be newly encoded')
            if ref.sha256 not in self.key_cache:
                rendered=backbone.tokenizer.apply_chat_template([{'role':'user','content':ref.content.decode()}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=backbone.tokenizer.encode(rendered,add_special_tokens=False)
                self.key_cache[ref.sha256]=reference.pooler(reference.annotation_states(ids),*SLOTS[ref.role])
                self.stats['reference_encodings']+=1;self.stats['reference_tokens']+=len(ids)
            else:self.stats['reference_cache_hits']+=1
            return self.key_cache[ref.sha256].clone()

    return FrozenEncoder()
