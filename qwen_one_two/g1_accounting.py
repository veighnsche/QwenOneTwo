"""Model-free G1 ledger, bit encoding and ordered scalar budget arithmetic."""
from dataclasses import dataclass, field
import math

class AdmissionError(RuntimeError):
    pass


@dataclass
class Ledger:
    prompt: int
    budget: int = 128
    answer: int = 16
    b: int = 0
    a: int = 0
    h: int = 0
    attempts: int = 0
    dispatched: int = 0
    writer: int = 0
    records: int = 0
    holds: int = 0
    released: int = 0
    phase: int = 0
    consumed: bool = False
    consumed_count: int = 0
    cancelled: bool = False
    events: list = field(default_factory=list)

    @property
    def position(self):
        return self.prompt+self.b+self.a+self.h

    def commit(self,label,kind,count=1):
        if self.cancelled:
            raise RuntimeError('post-cancellation model work')
        if kind=='B': self.b+=count
        elif kind=='A': self.a+=count
        elif kind=='H': self.h+=count
        else: raise ValueError(kind)
        assert self.b<=self.budget and self.a<=self.answer and self.h<=64
        assert self.position<=256
        for i in range(count):
            self.events.append({'label':label,'kind':kind,'position':self.position-count+i+1})

    def counters(self):
        import torch  # only this CUDA feature constructor requires Torch
        return torch.tensor([(self.budget-self.b)/131072,(self.answer-self.a)/262144,
                             (64-self.h)/64,(262144-self.position)/262144,
                             (32-self.writer)/32,(16-self.attempts)/16,
                             (32-self.records)/32,self.phase],device='cuda',dtype=torch.float32)

    def admit(self,q,rmax,request_handles):
        # Independent position, Ref and handle envelopes. G1's K contains no
        # compatible existing records outside the rendered request table.
        assert self.records+1+2<=32
        assert request_handles+2<=256
        self.records+=1  # Snapshot publication at sealing is not refunded on denial.
        if self.budget-self.b<q+rmax:
            raise AdmissionError('one-position-short or exhausted B reservation')
        if 262144-self.position-(self.answer-self.a)-(64-self.h)<q+rmax:
            raise AdmissionError('context reservation')
        self.dispatched+=1
        self.holds=q+rmax

    def finish_return(self,q,r):
        assert not self.consumed
        assert q+r<=self.holds
        self.released=self.holds-q-r
        self.holds=0
        self.records+=1
        self.consumed=True
        self.consumed_count+=1


def bit_fields(values):
    bits=[0]*128
    for start,width,value in values:
        if not 0<=int(value)<2**width:
            raise ValueError((start,width,value))
        for j in range(width): bits[start+j]=(int(value)>>(width-j-1))&1
    return bits


def budget(raw):
    if not math.isfinite(raw): raise ValueError('nonfinite startup score')
    s=min(1.,max(0.,raw));u=s*s;v=3.*u;w=2.*u;z=w*s
    return math.floor((v-z)*131072.+.5)


