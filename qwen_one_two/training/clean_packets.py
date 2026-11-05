"""Build clean component requests/returns with the production codec renderer.

This is offline fixture construction, never a production second prefill. Labels
remain external; supplied reading returns are explicitly component inputs.
"""
import json
import torch
from ..codec.records import Semantic, Span, Attribute, Value, Reason
from ..codec.schema import Kind, Port, Status, Role, ValueKind, Mode, parse_numeric
from ..codec.runtime import Runtime
from .inputs import check, digest, canonical, REQUEST_KINDS


class CleanPackets:
    def __init__(self, encoded, view, provenance):
        excerpts,model_bytes = view.materialize(provenance)
        check(encoded['input_hash'] == digest(model_bytes), 'clean view identity mismatch')
        check(len(encoded['sources']) == len(excerpts) and len(encoded['slots']) == len(view.references), 'clean artifact counts')
        original = encoded['registry']
        check(set(original.refs()) == set(encoded['sources']), 'unclean initial registry contents')
        self.registry = original.fork()
        self.sources,self.slots,self.view = encoded['sources'],tuple(x.clone() for x in encoded['slots']),view
        self.attributes = {}
        self.audit = encoded['audit']
        check(len(self.audit) == len(self.sources)+len(self.slots), 'clean lineage count')
        for index,ref in enumerate(self.sources):
            source = self.registry.get(ref,Kind.SOURCE)
            audit = self.audit[index]
            check(source.text == excerpts[index] and digest(source.text) == audit['input_hash'], 'source content drift')
            check(source.offsets == audit['offsets'] and self.tensor_hash(source.states) == audit['tensor_hash'], 'source artifact drift')
        for index,payload in enumerate(self.slots):
            audit = self.audit[len(self.sources)+index]
            check(audit['input_hash'] == view.references[index].sha256 and
                  self.tensor_hash(payload) == audit['tensor_hash'], 'reference artifact drift')
        self.runtime = None

    @staticmethod
    def tensor_hash(value):
        check(value.is_cuda and value.dtype == torch.float32, 'clean FP32 CUDA tensor')
        return digest(value.detach().contiguous().cpu().numpy().tobytes())

    def slots_for(self,index,role):
        check(type(index) is int and 0 <= index < len(self.slots), 'clean reference index')
        check(self.view.references[index].role == role, 'clean reference role mismatch')
        check(self.tensor_hash(self.slots[index]) == self.audit[len(self.sources)+index]['tensor_hash'], 'clean slots changed')
        return self.slots[index]

    def span(self,source,start,count,clip_start=0,clip_end=None):
        check(type(source) is int and 0 <= source < len(self.sources), 'clean local source index')
        span = Span(self.sources[source],start,count,clip_start,clip_end)
        self.registry.span(span)
        return span

    def source_attribute(self,key,span,kind):
        """Literal source-backed exact values; no outcome-derived attributes."""
        check(span.source in self.sources, 'foreign source attribute')
        _,literal,_,_ = self.registry.span(span)
        key_slots = self.slots_for(key,'key')
        if kind in (ValueKind.INT64,ValueKind.DECIMAL):
            value = Value(kind,True,parse_numeric(literal,kind),Mode.PARSE,span.source,parsed_span=span)
        elif kind == ValueKind.SOURCE_STRING:
            value = Value(kind,True,span,Mode.CONSTRUCT,span.source)
        elif kind == ValueKind.UTF8:
            value = Value(kind,True,literal,Mode.CONSTRUCT,span.source)
        elif kind == ValueKind.BOOLEAN:
            check(literal in (b'true',b'false'), 'exact Boolean literal required')
            value = Value(kind,True,literal == b'true',Mode.CONSTRUCT,span.source)
        else:
            raise ValueError('reference attributes require independently reviewed local object bindings')
        attribute = Attribute(key_slots.clone(),value)
        self.registry.attribute(attribute)
        self.attributes[id(attribute)] = (key,value)
        return attribute

    def reference_attribute(self,key,referent,span):
        """A local identity whose neutral literal is present in the clean source."""
        check(span.source in self.sources, 'foreign reference source')
        _,literal,_,_ = self.registry.span(span)
        meaning = self.slots_for(referent,'target')
        annotation = json.loads(self.view.references[referent].content)['fields']['referent']
        check(annotation.encode() == literal, 'reference literal/neutral identity mismatch')
        ref = self.registry.publish(Kind.TARGET,Semantic(meaning))
        value = Value(ValueKind.REFERENCE,True,ref,Mode.CONSTRUCT,span.source)
        attribute = Attribute(self.slots_for(key,'key').clone(),value)
        self.registry.attribute(attribute)
        self.attributes[id(attribute)] = (key,value)
        return attribute

    def quoted_utf8_attribute(self,key,span):
        check(span.source in self.sources, 'foreign UTF8 source')
        _,literal,_,_ = self.registry.span(span)
        value = json.loads(literal)
        check(isinstance(value,str), 'quoted UTF8 string required')
        body = Value(ValueKind.UTF8,True,value.encode(),Mode.CONSTRUCT,span.source)
        attribute = Attribute(self.slots_for(key,'key').clone(),body)
        self.registry.attribute(attribute)
        self.attributes[id(attribute)] = (key,body)
        return attribute

    def absent_attribute(self,key,kind,provenance,node):
        # Declared absence is metadata, never a fabricated source span/value.
        provenance.permit((node,),REQUEST_KINDS,self.view.permitted_nodes)
        check(provenance.nodes[node].content_hash == digest(canonical({'kind':int(kind),'present':False})), 'absence provenance')
        body = Value(kind,False)
        attribute = Attribute(self.slots_for(key,'key').clone(),body)
        self.registry.attribute(attribute)
        self.attributes[id(attribute)] = (key,body)
        return attribute

    def request(self,port,question,target,*,evidence=(),candidates=(),rubric=(),direction=1,
                target_attributes=(),candidate_attributes=(),session=None):
        check(self.runtime is None, 'one immutable clean request per construction')
        check(self.view.operation in (None,{Port.NOUL:'Noul',Port.CHOICE:'Choice',Port.SCORE:'Score'}[port]), 'clean operation mismatch')
        check(all(isinstance(span,Span) and span.source in self.sources for span in evidence), 'clean evidence scope')
        # No incoming state, draft, catalogue or registry from the full context.
        if session is None:
            rt = Runtime(self.registry.run,0,512,128)
            rt.registry = self.registry
            rt.startup(); rt.begin(port)
        else:
            rt=session.runtime
            check(rt.registry.run==self.registry.run and not rt.registry.refs() and rt.state=='Native',
                  'clean component requires a fresh session registry')
            rt.registry=self.registry
            writer=session.begin_writer(port)
        def validate_attrs(attributes):
            for attribute in attributes:
                check(id(attribute) in self.attributes, 'attribute not constructed from this clean source view')
                key,expected = self.attributes[id(attribute)]
                check(attribute.value == expected and torch.equal(attribute.keys,self.slots_for(key,'key')), 'clean attribute drift')
                self.registry.attribute(attribute)
            return tuple(attributes)
        q = rt.publish(Kind.QUESTION,Semantic(self.slots_for(question,'question')))
        t = rt.publish(Kind.TARGET,Semantic(self.slots_for(target,'target'),validate_attrs(target_attributes)))
        refs = []
        check(not candidate_attributes or len(candidate_attributes) == len(candidates), 'candidate attribute count')
        for index,slot in enumerate(candidates):
            attrs = candidate_attributes[index] if candidate_attributes else ()
            refs.append(rt.publish(Kind.ALTERNATIVE,Semantic(self.slots_for(slot,'alternative'),validate_attrs(attrs))))
        rubric_ref = None
        if rubric:
            check(len(rubric) in (4,5), 'rubric fields')
            payload = torch.cat([self.slots_for(index,role) for index,role in zip(rubric,('dimension','conditions','zero','one','midpoint'))])
            if len(rubric) == 4:
                payload = torch.cat((payload,payload.new_zeros(2,1024)))
            rubric_ref = rt.publish(Kind.RUBRIC,Semantic(payload,direction=direction,midpoint=len(rubric)==5))
        if session is None:
            rt.writer_position(); rt.writer_position()
        else:
            from dataclasses import replace
            writer.draft=replace(writer.draft,question=q,target=t,evidence=tuple(evidence),
                                 candidates=tuple(refs),rubric=rubric_ref)
            session.step(writer.tick_vector(session.hidden[0,-1])[None],'R','component_finish',False)
        packet = rt.seal(q,t,tuple(evidence),rubric_ref,tuple(refs))
        self.runtime = rt
        return packet

    def supplied_return(self,status,*,scalar=None,selected_index=None,reason=None):
        check(self.view.purpose == 'reader' and self.runtime is not None, 'supplied return only in clean reader construction')
        rt = self.runtime
        check(rt.state == 'Evaluating', 'return already constructed')
        # Offline construction has not run a request forward. A real component
        # Session has already charged/evaluated it; never debit it twice.
        if rt.call.q_remaining:
            check(rt.call.q_remaining==len(rt.call.request),'partially evaluated component request')
            rt.semantic_positions(len(rt.call.request))
        if reason is not None:
            from .episodes import STATUS_NAMES
            check(json.loads(self.view.references[reason].content)['fields']['status'] == STATUS_NAMES[status], 'Reason/carrier mismatch')
        payload = None if reason is None else Reason(self.slots_for(reason,'reason'))
        # Runtime validates status/value presence, bank selection, immutable
        # binding and every completion reservation; no hand-built result bypass.
        return rt.prepare_result(rt.call.snapshot,status,scalar=scalar,
                                 selected_index=selected_index,reason=payload)

    def encode(self,codec,packet):
        """Encode authoritative bindings and return external construction audit."""
        from ..codec.render import Renderer
        from .reference import state_fingerprint
        rt = self.runtime
        check(rt is not None and packet.snapshot == rt.call.snapshot, 'foreign clean packet')
        if packet.result is None:
            authoritative = Renderer(self.registry).request(rt.call.snapshot)
        else:
            check(packet.result == rt.call.result, 'foreign clean return')
            authoritative = self.registry.get(rt.call.result,Kind.RESULT).presentation
        encoded = codec.encode(authoritative)
        return encoded, {'construction':'clean-packets-v1','purpose':self.view.purpose,
                         'source_and_reference_lineage':self.audit,
                         'transformations':['permitted_view','grounded_exact_fields','production_renderer','codec.encode'],
                         'codec_state_hash':state_fingerprint(codec.state_dict()),
                         'tensor_hash':self.tensor_hash(encoded),'positions':len(authoritative),
                         'supplied_component_return':packet.result is not None,
                         'runtime':rt.summary()}
