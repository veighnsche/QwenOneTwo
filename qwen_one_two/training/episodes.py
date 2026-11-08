"""Bind supervision to actual packets and validate complete one-cache episodes."""
from dataclasses import dataclass
from .inputs import check, Exposure
from .losses import ExampleLosses, Eligibility, ce, binary, permitted_set, regression, alignment
from ..codec.schema import Port, Status, Kind

STATUS_NAMES = {Status.ASSESSED:'Assessed', Status.INSUFFICIENT:'Insufficient',
                Status.CONFLICTING:'Conflicting', Status.UNRESOLVED:'Unresolved',
                Status.NO_FEASIBLE_CHOICE:'NoFeasibleChoice'}
PORT_NAMES = {Port.NOUL:'Noul', Port.CHOICE:'Choice', Port.SCORE:'Score'}


@dataclass(frozen=True)
class DecisionLabels:
    """Reviewed labels for the exact actual sealed snapshot, not withheld truth.

    The data validator must establish semantics/review before making this object;
    a matching identity alone does not constitute semantic approval.
    """
    snapshot: object
    status: Status
    noul: float | None = None
    choice: tuple = ()  # stable refs from this bank, never opaque UUID classes
    score_point: float | None = None
    score_interval: tuple | None = None
    feasibility: tuple = ()  # (stable candidate Ref, class 0..3)
    reason_slots: object = None


def decision_losses(registry, actual_snapshot, actual_status, evaluated, labels):
    book = ExampleLosses()
    if labels is None or labels.snapshot != actual_snapshot:
        return book.finish(), {'supported_packet':False,'reason':'missing or divergent binding'}
    snapshot = registry.get(actual_snapshot,Kind.SNAPSHOT)
    check(labels.status in STATUS_NAMES and actual_status in STATUS_NAMES, 'semantic status')
    check(labels.status != Status.NO_FEASIBLE_CHOICE or snapshot.port == Port.CHOICE, 'Choice-only reference status')
    masks = Eligibility(True,STATUS_NAMES[labels.status],STATUS_NAMES[actual_status],PORT_NAMES[snapshot.port],
                        reason_supported=labels.reason_slots is not None).mask()
    book.add('status','status',ce(evaluated['status_logits'],int(labels.status)-1))
    candidates = () if snapshot.bank is None else registry.get(snapshot.bank,Kind.BANK).candidates
    if masks['feasibility']:
        check(len({ref for ref,_ in labels.feasibility}) == len(labels.feasibility), 'duplicate feasibility target')
        for ref,value in labels.feasibility:
            check(ref in candidates and type(value) is int and 0 <= value < 4, 'feasibility binding/class')
            index = candidates.index(ref)
            book.add('feasibility',str(index),ce(evaluated['feasibility_logits'][index],value))
    if masks['noul'] and labels.noul is not None:
        book.add('noul','truth',binary(evaluated['raw'],labels.noul))
    if masks['choice'] and labels.choice:
        check(len(set(labels.choice)) == len(labels.choice) and all(ref in candidates for ref in labels.choice), 'absent/duplicate acceptable candidate')
        book.add('choice','selection',permitted_set(evaluated['pointer_logits'],[candidates.index(ref) for ref in labels.choice]))
    if masks['score'] and (labels.score_point is not None or labels.score_interval is not None):
        book.add('score','raw',regression(evaluated['raw'],point=labels.score_point,interval=labels.score_interval))
    if masks['reason']:
        book.add('alignment','reason',alignment(evaluated['reason_slots'],labels.reason_slots),role='reason')
    return book.finish(), {'supported_packet':True,'masks':masks,'eligible':sorted(book.fields)}


def collect_session(window, session, losses, exposure):
    """Validate the real Session, then collect its complete trajectory graph.

    Cancelled/failed/overlength jobs remain scheduled failed exposures without
    truncated-prefix gradients. No extra prefill or detached replay is used.
    """
    check(isinstance(exposure,Exposure), 'explicit exposure metadata required')
    check(not getattr(session,'episode_report',{}).get('diagnostic_reference',False),
          'diagnostic reference episodes cannot enter optimizer accumulation')
    from .receiver_profiles import LEGACY
    from ..codec.layout import DIRECT
    report=getattr(session,'episode_report',{})
    if report.get('input_profile',LEGACY)!=LEGACY or getattr(session.runtime,'profile',None)==DIRECT:
        from .component_admission import PreparedReceiver
        admission=getattr(session,'component_admission',None)
        check(exposure.branch=='isolated_receiver',
              'new-profile optimization requires explicit prepared receiver admission; native semantics stay masked')
        # A failed attempt with no gradients still occupies its scheduled slot.
        # Only a completed optimization graph requires semantic admission.
        if losses or session.runtime.terminal not in ('Failed','Cancelled'):
            check(isinstance(admission,PreparedReceiver),'new-profile optimization requires prepared receiver admission')
            admission.validate(session.runtime)
            check(report.get('component_admission')==admission.receipt(),'experimental admission telemetry drift')
    exposure.validate()
    summary = session.summary()
    rt = session.runtime
    if window.phase in {'transition','free_joint'}:
        check(exposure.budget_override is None and summary['override'] is None, 'override after guided phase')
    # A4 retains an isolated decision-replay exposure in free-control windows.
    # Its reviewed component packet is distinct from forcing the native policy
    # in a joint/direct trajectory; preserve both exposure labels explicitly.
    if window.phase == 'free_joint' and exposure.branch != 'isolated_receiver':
        check(not exposure.decision_forcing, 'forced free-control decisions')
    check(exposure.budget_override == summary['override'], 'override telemetry mismatch')
    check(summary['prefills'] == 1 and summary['lineage'] == 1, 'one prefill/cache lineage')
    check(summary['total'] == sum(summary[k] for k in ('P','B','A','H')) == session.state.position, 'exact P+B+A+H')
    check(rt.state == 'Terminal', 'backward before episode termination')
    check(summary['held_positions'] == 0 and all(c.handle_hold == 0 and c.record_hold == 0 for c in rt.calls), 'outstanding completion holds')
    check(all(c.consumed for c in rt.calls if c.state == 'Consumed'), 'consumption state')
    completed = summary['terminal'] in {'Completed','Limited'}
    dispatched = [c for c in rt.calls if c.snapshot is not None and not c.violations]
    check(len(dispatched) == summary['dispatched'] and sum(c.consumed for c in dispatched) == summary['consumed'], 'dispatch/consumption count')
    check(all(c.consumed or c.outcome for c in dispatched), 'unsettled dispatched call')
    window.add_episode(losses,complete=completed,positions=summary['total'],
                       failure=None if completed else summary['terminal'])
    return {'summary':summary,'exposure':exposure.__dict__.copy(),'accepted':completed and summary['total'] <=
            (1024 if window.phase in {'guided_joint','transition','free_joint'} else 512)}
