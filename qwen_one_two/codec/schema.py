"""Exact, model-free B1 identifiers, metadata and bounded typed atoms."""
from dataclasses import dataclass
from enum import IntEnum
import re


class Invalid(ValueError):
    pass


class Port(IntEnum):
    NONE = 0
    NOUL = 1
    CHOICE = 2
    SCORE = 3
    BUDGET = 4


class Status(IntEnum):
    NONE = 0
    ASSESSED = 1
    INSUFFICIENT = 2
    CONFLICTING = 3
    UNRESOLVED = 4
    NO_FEASIBLE_CHOICE = 5


class Kind(IntEnum):
    NONE = 0
    SOURCE = 1
    TARGET = 2
    QUESTION = 3
    ALTERNATIVE = 4
    BANK = 5
    RUBRIC = 6
    RESULT = 7
    REASON = 8
    SNAPSHOT = 9


class ValueKind(IntEnum):
    NONE = 0
    BOOLEAN = 1
    INT64 = 2
    DECIMAL = 3
    REFERENCE = 4
    SOURCE_STRING = 5
    UTF8 = 6
    CONTINUOUS = 7


class Mode(IntEnum):
    NONE = 0
    COPY = 1
    PARSE = 2
    CONSTRUCT = 3


class Role(IntEnum):
    PAD = 0
    REQUEST = 1
    QUESTION = 2
    TARGET_DESC = 3
    TARGET = 4
    SOURCE_DESC = 5
    SOURCE = 6
    HISTORY_DESC = 7
    RUBRIC_DESC = 8
    DIMENSION = 9
    CONDITIONS = 10
    ZERO = 11
    ONE = 12
    MIDPOINT = 13
    CANDIDATE_DESC = 14
    CANDIDATE = 15
    ATTRIBUTE_DESC = 16
    KEY = 17
    BOOLEAN = 18
    INT64 = 19
    COEFFICIENT = 20
    EXPONENT = 21
    REFERENCE = 22
    ATTRIBUTE_SOURCE = 23
    UTF8 = 24
    EVALUATE = 25
    RESULT = 26
    NOUL = 27
    SCORE = 28
    REASON_DESC = 29
    REASON = 30
    RETURN_END = 31
    BUDGET_HEADER = 32
    BUDGET = 33
    BUDGET_END = 34
    WRITER_START = 35
    WRITER_TICK = 36
    CONSUME = 37
    CLOSE = 38
    LIMIT = 39
    TERMINATE = 40


HEADERS = {Role.REQUEST, Role.RESULT, Role.BUDGET_HEADER}
DESCRIPTORS = {Role.TARGET_DESC, Role.SOURCE_DESC, Role.HISTORY_DESC, Role.RUBRIC_DESC,
               Role.CANDIDATE_DESC, Role.ATTRIBUTE_DESC, Role.EVALUATE, Role.REASON_DESC,
               Role.RETURN_END, Role.BUDGET_END}


def require(condition, message):
    if not condition:
        raise Invalid(message)


def integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, name)
    return value


def bits(fields, width=128):
    result = [0] * width
    occupied = set()
    for start, size, value in fields:
        integer(value, 0, (1 << size) - 1, 'field range')
        require(0 <= start < start + size <= width, 'field bounds')
        for j in range(size):
            require(start + j not in occupied, 'overlapping fields')
            occupied.add(start + j)
            result[start + j] = (value >> (size - 1 - j)) & 1
    return tuple(result)


@dataclass(frozen=True)
class Metadata:
    port: Port
    status: Status = Status.NONE
    value_kind: ValueKind = ValueKind.NONE
    mode: Mode = Mode.NONE
    kind: Kind = Kind.NONE
    present: bool = False
    owner: int | None = None
    item: int | None = None
    primary: int | None = None
    secondary: int | None = None
    unit: int | None = None
    question: int | None = None
    target: int | None = None
    bank: int | None = None
    rubric: int | None = None
    midpoint: bool = False
    historical: bool = False
    sealed: bool = True
    following: int = 0
    evidence: int = 0
    candidates: int = 0
    children: int = 0
    attempt: int = 0
    complete: int = 0
    direction: int = 0
    parameter_a: int = 0
    parameter_b: int = 0
    profile: str = 'b1-full-v1-compensated'
    field_lengths: tuple = ()

    def encode(self, role):
        from .layout import DIRECT, valid_profile, semantic_lengths, is_direct, field_limit, wire_version
        role = Role(role)
        valid_profile(self.profile)
        direct=is_direct(self.profile)
        length=0
        if direct and role in (Role.REQUEST,Role.RESULT,Role.TARGET_DESC,Role.CANDIDATE_DESC,Role.ATTRIBUTE_DESC):
            require(type(self.field_lengths) is tuple and len(self.field_lengths)==1,'direct metadata field length')
            length=integer(self.field_lengths[0],1,field_limit(self.profile),'direct metadata length')
        elif direct and role==Role.RUBRIC_DESC:
            semantic_lengths(Kind.RUBRIC,self.profile,self.field_lengths,midpoint=self.midpoint)
        else:require(self.field_lengths==(),'inapplicable metadata lengths')
        require(role in HEADERS | DESCRIPTORS, 'metadata role')
        Port(self.port); Status(self.status); ValueKind(self.value_kind); Mode(self.mode); Kind(self.kind)
        for name in ('owner', 'primary', 'secondary', 'unit', 'question', 'target', 'bank', 'rubric'):
            v = getattr(self, name)
            if v is not None:
                integer(v, 0, 255, name)
        if self.item is not None:
            integer(self.item, 0, 255, 'item')
        integer(self.following, 0, 511, 'following')
        integer(self.direction, 0, 2, 'direction')
        integer(self.children, 0, 8, 'children')
        flags = (self.present, self.owner is not None, self.item is not None,
                 self.primary is not None, self.secondary is not None, self.unit is not None,
                 self.question is not None, self.target is not None, self.bank is not None,
                 self.rubric is not None, self.midpoint, self.historical, self.sealed)
        require(all(type(v) is bool for v in flags), 'flag type')
        f = [(0, 4, wire_version(self.profile)), (4, 3, int(self.port)), (7, 3, int(self.status)),
             (10, 3, int(self.value_kind)), (13, 2, int(self.mode)), (15, 4, int(self.kind)),
             *[(19 + i, 1, int(v)) for i, v in enumerate(flags)],
             (32, 8, self.owner or 0), (40, 8, self.item or 0), (48, 9, self.following), (57, 7, length)]
        if role in HEADERS:
            require(self.primary is self.secondary is self.unit is self.item is None, 'header descriptor refs')
            require(self.parameter_a == self.parameter_b == 0, 'header parameters')
            integer(self.evidence, 0, 8, 'evidence'); integer(self.candidates, 0, 16, 'candidates')
            integer(self.attempt, 0, 16, 'attempt'); integer(self.complete, 1, 256, 'complete')
            require(self.following == self.complete - 1, 'header length')
            f += [(64, 8, self.question or 0), (72, 8, self.target or 0),
                  (80, 8, self.bank or 0), (88, 8, self.rubric or 0),
                  (96, 4, self.evidence), (100, 5, self.candidates), (105, 4, self.children),
                  (109, 5, self.attempt), (114, 9, self.complete), (123, 2, self.direction)]
        else:
            require(self.question is self.target is self.bank is self.rubric is None, 'descriptor header refs')
            require(self.evidence == self.candidates == self.attempt == self.complete == 0, 'descriptor header fields')
            f += [(64, 8, self.primary or 0), (72, 8, self.secondary or 0),
                  (80, 8, self.unit or 0), (88, 4, self.children), (92, 2, self.direction)]
            f += ([(96+6*i,6,n) for i,n in enumerate(self.field_lengths)] if direct and role==Role.RUBRIC_DESC else
                  [(96,16,self.parameter_a),(112,16,self.parameter_b)])
        if role not in (Role.REQUEST, Role.RESULT, Role.BUDGET_HEADER, Role.ATTRIBUTE_DESC):
            require(self.status == Status.NONE and self.value_kind == ValueKind.NONE and self.mode == Mode.NONE,
                    'inapplicable status/value/mode')
        if role not in (Role.REQUEST, Role.RESULT, Role.RUBRIC_DESC):
            require(not self.midpoint and not self.direction, 'inapplicable rubric flags')
        if role not in (Role.SOURCE_DESC, Role.ATTRIBUTE_DESC):
            require(self.parameter_a == self.parameter_b == 0, 'inapplicable parameters')
        if role in (Role.REQUEST,Role.RESULT):
            require(self.owner is not None and self.question is not None and self.target is not None,'required header bindings')
            require(self.port in (Port.NOUL,Port.CHOICE,Port.SCORE),'operation header')
            require((self.bank is not None)==(self.port==Port.CHOICE) and
                    (self.rubric is not None)==(self.port==Port.SCORE),'operation header references')
            require(1<=self.attempt<=16 and self.sealed,'sealed header')
            require((1<=self.candidates<=16) if self.port==Port.CHOICE else self.candidates==0,'bank size')
            require(self.direction in (1,2) if self.port==Port.SCORE else self.direction==0,'header scale')
            if role==Role.REQUEST:
                require(self.kind==Kind.SNAPSHOT and not self.present and not self.historical and
                        self.status==Status.NONE and self.value_kind==ValueKind.NONE and self.mode==Mode.NONE,'request fields')
            else:
                require(self.kind==Kind.RESULT and self.status!=Status.NONE and self.evidence==self.children==0,'result fields')
                require(self.status!=Status.NO_FEASIBLE_CHOICE or self.port==Port.CHOICE,'Choice-only status')
                expected=(ValueKind.REFERENCE if self.port==Port.CHOICE else ValueKind.CONTINUOUS) if self.status==Status.ASSESSED else ValueKind.NONE
                require(self.present==(self.status==Status.ASSESSED) and self.value_kind==expected and self.mode==Mode.NONE,'result presence')
        elif role==Role.BUDGET_HEADER:
            require(self.port==Port.BUDGET and self.kind==Kind.NONE and self.owner is None and
                    self.question is self.target is self.bank is self.rubric is None and
                    self.status==Status.NONE and self.value_kind==ValueKind.CONTINUOUS and self.mode==Mode.NONE and
                    self.present and self.complete==3 and self.following==2 and not self.sealed and not self.historical and
                    self.evidence==self.candidates==self.children==self.attempt==0,'budget header')
        else:
            kinds={Role.TARGET_DESC:Kind.TARGET,Role.SOURCE_DESC:Kind.SOURCE,Role.HISTORY_DESC:Kind.RESULT,
                   Role.RUBRIC_DESC:Kind.RUBRIC,Role.CANDIDATE_DESC:Kind.ALTERNATIVE,
                   Role.EVALUATE:Kind.SNAPSHOT,Role.REASON_DESC:Kind.REASON,Role.RETURN_END:Kind.RESULT,
                   Role.BUDGET_END:Kind.NONE}
            if role!=Role.ATTRIBUTE_DESC:require(self.kind==kinds[role],'descriptor owner kind')
            else:require(self.kind in (Kind.TARGET,Kind.ALTERNATIVE,Kind.REASON),'attribute owner kind')
            require((self.owner is None)==(role==Role.BUDGET_END),'descriptor owner presence')
            require((self.item is not None)==(role in (Role.SOURCE_DESC,Role.HISTORY_DESC,Role.CANDIDATE_DESC)), 'descriptor item applicability')
            if role not in (Role.TARGET_DESC,Role.CANDIDATE_DESC,Role.REASON_DESC):require(self.children==0,'descriptor children')
            if role not in (Role.CANDIDATE_DESC,Role.ATTRIBUTE_DESC,Role.REASON_DESC):require(self.primary is None,'descriptor primary')
            if role not in (Role.HISTORY_DESC,Role.ATTRIBUTE_DESC):require(self.secondary is None,'descriptor secondary')
            if role!=Role.ATTRIBUTE_DESC:require(self.unit is None,'descriptor unit')
            if role in (Role.EVALUATE,Role.RETURN_END,Role.BUDGET_END):require(self.following==0 and not self.present,'boundary fields')
            elif role!=Role.ATTRIBUTE_DESC:require(self.present,'record descriptor presence')
            if role==Role.SOURCE_DESC:require(1<=self.following<=32,'source length')
            if role==Role.HISTORY_DESC:require(1<=self.following<=64 and self.secondary is not None,'historical binding/length')
            if role==Role.RUBRIC_DESC:require(self.following==(sum(self.field_lengths) if direct else 10) and self.direction in (1,2),'rubric descriptor')
            if role==Role.ATTRIBUTE_DESC:
                require(self.value_kind in tuple(ValueKind)[1:7] and self.status==Status.NONE,'attribute kind')
                require((self.secondary is not None)==(self.mode==Mode.COPY),'copy binding flag')
                if not self.present:require(self.following==(length if direct else 2) and self.mode==Mode.NONE and self.primary is self.secondary is None,'absent attribute')
                else:require(self.mode!=Mode.NONE,'present creation mode')
                if self.mode==Mode.PARSE:require(self.value_kind in (ValueKind.INT64,ValueKind.DECIMAL) and self.primary is not None,'numeric parse metadata')
                if self.value_kind==ValueKind.UTF8:require(self.parameter_a<=128 and self.parameter_b==0,'UTF8 descriptor')
                elif self.value_kind!=ValueKind.SOURCE_STRING:require(self.parameter_a==self.parameter_b==0,'attribute parameters')
        return bits(f)


@dataclass(frozen=True)
class Atom:
    role: Role
    data: bytes
    coefficient: int | None = None
    exponent: int = 0

    def __post_init__(self):
        require(type(self.data) is bytes and len(self.data) == 8, 'atom bytes')
        require(self.role in (Role.BOOLEAN, Role.INT64, Role.COEFFICIENT, Role.EXPONENT,
                              Role.REFERENCE, Role.UTF8), 'atom role')
        if self.coefficient is not None:
            integer(self.coefficient, -(1 << 63), (1 << 63) - 1, 'coefficient')
            integer(self.exponent, -18, 18, 'exponent')


def int_bytes(n):
    integer(n, -(1 << 63), (1 << 63) - 1, 'int64')
    return n.to_bytes(8, 'big', signed=True)


def encode_atoms(kind, value, handle=None):
    kind = ValueKind(kind)
    if kind == ValueKind.BOOLEAN:
        require(type(value) is bool, 'boolean')
        return (Atom(Role.BOOLEAN, bytes(7) + bytes([int(value)])),)
    if kind == ValueKind.INT64:
        return (Atom(Role.INT64, int_bytes(value), value),)
    if kind == ValueKind.DECIMAL:
        require(type(value) is tuple and len(value) == 2, 'decimal pair')
        c, e = value
        data = int_bytes(c); integer(e, -18, 18, 'decimal exponent')
        return (Atom(Role.COEFFICIENT, data, c, e),
                Atom(Role.EXPONENT, bytes(7) + bytes([e & 255]), c, e))
    if kind == ValueKind.REFERENCE:
        integer(handle, 0, 255, 'reference handle')
        return (Atom(Role.REFERENCE, bytes(7) + bytes([handle])),)
    if kind == ValueKind.UTF8:
        require(type(value) is bytes and len(value) <= 128, 'UTF8 length')
        try:
            value.decode('utf-8', errors='strict')
        except UnicodeError as exc:
            raise Invalid('invalid UTF8') from exc
        return tuple(Atom(Role.UTF8, value[i:i + 8].ljust(8, b'\0')) for i in range(0, len(value), 8))
    raise Invalid('not an atom type')


def decode_atoms(kind, atoms, byte_length=0):
    kind = ValueKind(kind)
    if kind == ValueKind.UTF8:
        integer(byte_length, 0, 128, 'UTF8 length')
        require(len(atoms) == (byte_length + 7) // 8, 'UTF8 atom count')
        require(all(x.role == Role.UTF8 for x in atoms), 'UTF8 role')
        raw = b''.join(x.data for x in atoms)
        require(not any(raw[byte_length:]), 'UTF8 padding')
        value = raw[:byte_length]
        encode_atoms(kind, value)
        return value
    require(len(atoms) == (2 if kind == ValueKind.DECIMAL else 1), 'atom count')
    a = atoms[0]
    if kind == ValueKind.BOOLEAN:
        require(a.role == Role.BOOLEAN and a.data[:7] == bytes(7) and a.data[7] <= 1, 'Boolean padding')
        return bool(a.data[7])
    if kind == ValueKind.INT64:
        require(a.role == Role.INT64, 'int64 role')
        return int.from_bytes(a.data, 'big', signed=True)
    if kind == ValueKind.DECIMAL:
        require(a.role == Role.COEFFICIENT and atoms[1].role == Role.EXPONENT, 'decimal roles')
        require(atoms[1].data[:7] == bytes(7), 'decimal exponent padding')
        c = int.from_bytes(a.data, 'big', signed=True)
        e = int.from_bytes(atoms[1].data[7:], 'big', signed=True)
        encode_atoms(kind, (c, e))
        return c, e
    if kind == ValueKind.REFERENCE:
        require(a.role == Role.REFERENCE and a.data[:7] == bytes(7), 'reference padding')
        return a.data[7]
    raise Invalid('not an atom type')


def parse_numeric(literal, kind):
    """Parse exactly the learned-selected literal; never search source text."""
    require(type(literal) is bytes, 'literal bytes')
    pattern = rb'[+-]?[0-9]+' if kind == ValueKind.INT64 else rb'[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    require(re.fullmatch(pattern, literal) is not None, 'numeric literal')
    if kind == ValueKind.INT64:
        result = int(literal)
    else:
        require(kind == ValueKind.DECIMAL, 'numeric type')
        parts = re.split(rb'[eE]', literal)
        base = parts[0]; exponent = int(parts[1]) if len(parts) == 2 else 0
        fractional = len(base.split(b'.')[1]) if b'.' in base else 0
        result = (int(base.replace(b'.', b'')), exponent - fractional)
    encode_atoms(kind, result)
    return result
