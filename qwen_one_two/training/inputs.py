"""F7 canonical input builders and mechanical provenance checks.

These validate declared lineage, not the truth of prose or reviewer decisions.
Semantic review must still detect misattributed conclusions in authored fields.
"""
from dataclasses import dataclass
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Source:
    identity: str
    version: int
    content: bytes
    sha256: str

    def validate(self):
        check(bool(self.identity) and self.version > 0, 'source identity/version')
        check(digest(self.content) == self.sha256, 'source hash mismatch')
        self.content.decode('utf-8')


@dataclass(frozen=True)
class Binding:
    source: str
    version: int
    start: int
    end: int

    def extract(self, sources):
        source = sources[self.source]
        source.validate()
        check(source.version == self.version, 'source version mismatch')
        check(0 <= self.start < self.end <= len(source.content), 'source byte coordinates')
        # Both endpoints must be original UTF-8 boundaries, without normalization.
        source.content[:self.start].decode('utf-8')
        source.content[:self.end].decode('utf-8')
        return source.content[self.start:self.end]


@dataclass(frozen=True)
class Node:
    kind: str
    content_hash: str
    parents: tuple[str, ...] = ()
    binding: Binding | None = None


class Provenance:
    KINDS = {'request_semantics', 'source_fact', 'historical_result', 'reason',
             'current_outcome', 'withheld', 'approved_return'}

    def __init__(self, nodes, sources):
        self.nodes = dict(nodes)
        self.sources = dict(sources)
        for key, source in self.sources.items():
            check(key == source.identity, 'source identity mismatch')
            source.validate()
        for key in self.nodes:
            self.dependencies(key)

    def dependencies(self, key, active=frozenset()):
        check(key not in active and key in self.nodes, 'cyclic or missing provenance')
        node = self.nodes[key]
        check(node.kind in self.KINDS and len(node.content_hash) == 64, 'provenance kind/hash')
        check(all(c in '0123456789abcdef' for c in node.content_hash), 'invalid hash')
        if node.binding is not None:
            check(digest(node.binding.extract(self.sources)) == node.content_hash, 'bound content mismatch')
        if node.kind == 'source_fact':
            check(node.binding is not None, 'fact without source coordinates')
        found = {key}
        for parent in node.parents:
            found.update(self.dependencies(parent, active | {key}))
        return found

    def permit(self, keys, allowed, permitted_nodes=None):
        dependencies = set()
        for key in keys:
            dependencies.update(self.dependencies(key))
        check(bool(dependencies), 'missing provenance')
        check(all(self.nodes[k].kind in allowed for k in dependencies), 'forbidden input dependency')
        if permitted_nodes is not None:
            check(dependencies <= set(permitted_nodes), 'dependency outside permitted view')
        return tuple(sorted(dependencies))


# Each role is encoded separately. There is no whole-scenario serialization.
FIELDS = {
    'question': ('proposition', 'criterion', 'scope', 'qualifications', 'polarity', 'constraints'),
    'target': ('referent',),
    'alternative': ('identity', 'properties'),
    'dimension': ('meaning',), 'conditions': ('meaning',),
    'zero': ('meaning',), 'one': ('meaning',), 'midpoint': ('meaning',),
    'key': ('property', 'unit'),
    'reason': ('status', 'support'),
}
SLOTS = {'question': (0, 8), 'target': (8, 2), 'alternative': (10, 4),
         'dimension': (14, 2), 'conditions': (16, 2), 'zero': (18, 2),
         'one': (20, 2), 'midpoint': (22, 2), 'key': (24, 2), 'reason': (26, 4)}
REQUEST_KINDS = {'request_semantics', 'source_fact', 'historical_result'}
NONASSESSED = {'Insufficient', 'Conflicting', 'Unresolved', 'NoFeasibleChoice'}


@dataclass(frozen=True)
class Field:
    text: str
    provenance: str


@dataclass(frozen=True)
class ReferenceInput:
    role: str
    content: bytes
    dependencies: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]

    @property
    def sha256(self):
        return digest(self.content)


def reference_input(role, fields, provenance, *, permitted_nodes=None, operation=None):
    """Strict fields, byte-bound provenance, no outcome columns in the API.

    The caller selects one record/role, never passes a scenario to an encoder.
    Missing optional fields are omitted, not padded with labels or fake facts.
    """
    check(role in FIELDS and bool(fields), 'unknown or empty role')
    check(set(fields) <= set(FIELDS[role]), 'field outside role whitelist')
    if role == 'reason':
        check(set(fields) == {'status', 'support'}, 'Reason requires status and support')
        check(fields['status'].text in NONASSESSED, 'Reason requires non-assessed status')
        check(fields['status'].text != 'NoFeasibleChoice' or operation == 'Choice', 'Choice-only Reason')
    allowed = REQUEST_KINDS | ({'reason'} if role == 'reason' else set())
    keys = []
    for field in fields.values():
        check(isinstance(field, Field) and bool(field.text.strip()), 'empty/untyped canonical field')
        check(digest(field.text.encode()) == provenance.nodes[field.provenance].content_hash,
              'annotation bytes do not match provenance')
        keys.append(field.provenance)
    deps = provenance.permit(keys, allowed, permitted_nodes)
    return ReferenceInput(role, canonical({'role': role, 'fields': {k: fields[k].text for k in fields}}), deps,
                          tuple(sorted((name, field.provenance) for name, field in fields.items())))


def scenario_reference(row, role, record, provenance, **kwargs):
    """Explicit selection ignores every outcome/guidance/review column."""
    return reference_input(role, row['semantics'][role][record], provenance, **kwargs)


def validate_record_manifest(record, parents, *, for_training=False, for_release=False):
    """Mechanical lineage gate; never creates reviews or certifies semantics."""
    required = {'id', 'version', 'family', 'split', 'parents', 'origin', 'construction',
                'sources', 'labels', 'reviews', 'release'}
    check(required <= set(record), 'missing mandatory provenance')
    check(all(isinstance(record[k], str) and record[k].strip() for k in ('id', 'family'))
          and type(record['version']) is int and record['version'] > 0
          and record['split'] in {'train', 'development', 'locked'}, 'identity/split')
    check(isinstance(record['parents'], list) and len(set(record['parents'])) == len(record['parents'])
          and record['id'] not in record['parents'], 'duplicate or self parent')
    check(record['origin'] in {'authored', 'deterministic', 'model_proposal', 'replay', 'rollout_correction'}, 'origin')
    for parent_id in record['parents']:
        parent = parents[parent_id]
        check(parent['family'] == record['family'] and parent['split'] == record['split'], 'family split leakage')
    if record['origin'] == 'model_proposal':
        check({'checkpoint', 'prompt_hash', 'sampling', 'software', 'gpu', 'output_hash'} <= set(record.get('model_assistance', {})), 'missing model provenance')
    if record['origin'] == 'replay':
        check(bool(record.get('reuse_rights')), 'missing replay rights')
    if record['origin'] == 'rollout_correction':
        check({'checkpoint', 'trace_hash', 'forcing', 'substitution', 'override'} <= set(record.get('rollout', {})), 'missing rollout provenance')
    if for_training:
        check(record['split'] == 'train' and not record.get('design_excluded', False), 'excluded training material')
    if for_training or for_release:
        check(record['release'].get('state') == 'approved' and record['reviews'], 'unreviewed material')
        check(bool(record.get('author')), 'missing author identity')
        reviews = record['reviews']
        required_count = 2 if record['split'] == 'locked' or any(
            record.get(k, False) for k in ('pilot', 'subjective', 'new_rubric')) else 1
        check(len({r.get('reviewer') for r in reviews}) >= required_count, 'insufficient independent reviews')
        for review in reviews:
            check(bool(review.get('reviewer')) and review['reviewer'] != record['author'], 'self or missing review')
            check(review.get('kind') == 'human' and review.get('semantics_blind') is True
                  and review.get('outcome_independent') is True, 'review independence')
            check(review.get('decision') == 'approved' and bool(review.get('artifact_hash'))
                  and review.get('record_version') == record['version'], 'review version/artifact/decision')
        check(not record.get('quarantined') and not record.get('unresolved'), 'unresolved material')


@dataclass(frozen=True)
class Exposure:
    branch: str
    packet_origin: str = 'actual'
    native_teacher_forcing: bool = False
    outcome_bearing_prefix: bool = False
    decision_forcing: bool = False
    semantic_substitution: bool = False
    gold_return: bool = False
    budget_override: int | None = None

    def validate(self):
        check(self.branch in {'reference', 'native_writer', 'isolated_receiver', 'joint_receiver',
                              'isolated_reader', 'joint_reader', 'reconstruction', 'free_evaluation'}, 'branch')
        check(self.packet_origin in {'actual', 'reviewed_component'}, 'packet origin')
        check(not self.gold_return or self.branch == 'isolated_reader', 'gold return outside isolated reading')
        if self.branch.startswith('joint') or self.branch == 'free_evaluation':
            check(self.packet_origin == 'actual' and not self.semantic_substitution and not self.gold_return,
                  'joint execution requires actual slots and returns')
        check(not self.outcome_bearing_prefix or self.native_teacher_forcing, 'undeclared prefix exposure')
        if self.branch == 'free_evaluation':
            check(not any((self.native_teacher_forcing, self.outcome_bearing_prefix,
                           self.decision_forcing, self.semantic_substitution, self.gold_return))
                  and self.budget_override is None, 'guided input in free evaluation')
        check(self.budget_override is None or self.budget_override == 512, 'A4 override must be fixed 512')


BRANCH_FIELDS = {
    'native_writer': {'causal_history', 'evidence', 'draft', 'counters'},
    'isolated_receiver': {'request', 'evidence'},
    'joint_receiver': {'causal_history', 'request'},
    'isolated_reader': {'bound_return', 'permitted_prefix'},
    'joint_reader': {'causal_history', 'bound_return'},
    'reconstruction': {'latent', 'role', 'slot', 'presence', 'preceding_target_tokens'},
    'free_evaluation': {'prompt'},
}


def branch_input(exposure, **fields):
    """Whitelist branch payloads; supervision/audit columns stay external.

    Tensor/record provenance is checked when constructing those payloads. This
    boundary does not infer whether arbitrary caller-provided prose is truthful.
    """
    exposure.validate()
    check(exposure.branch in BRANCH_FIELDS, 'use role builder for reference inputs')
    check(set(fields) == BRANCH_FIELDS[exposure.branch], 'branch input whitelist')
    return dict(fields)
