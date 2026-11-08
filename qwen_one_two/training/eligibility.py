"""Model-free supervision masks, shared unchanged by CUDA loss construction."""
from dataclasses import dataclass
from .inputs import check


@dataclass(frozen=True)
class Eligibility:
    supported_packet: bool
    reference_status: str | None
    actual_status: str | None
    operation: str
    reason_supported: bool = False
    runtime_failure: bool = False

    def __post_init__(self):
        statuses = {'Assessed','Insufficient','Conflicting','Unresolved','NoFeasibleChoice',None}
        check(self.operation in {'Noul','Choice','Score'} and self.reference_status in statuses
              and self.actual_status in statuses, 'eligibility operation/status')
        check(self.operation == 'Choice' or 'NoFeasibleChoice' not in (self.reference_status,self.actual_status), 'Choice-only eligibility')

    def mask(self):
        semantic = self.supported_packet and not self.runtime_failure and self.reference_status is not None
        assessed = semantic and self.reference_status == 'Assessed'
        return {'status':semantic, 'noul':assessed and self.operation == 'Noul',
                'choice':assessed and self.operation == 'Choice',
                'score':assessed and self.operation == 'Score',
                'feasibility':semantic and self.operation == 'Choice',
                'reason':semantic and self.reason_supported and self.reference_status == self.actual_status
                         and self.actual_status in {'Insufficient','Conflicting','Unresolved','NoFeasibleChoice'}}
