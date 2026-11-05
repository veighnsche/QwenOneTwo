# Control flow v1: consumption, closure and termination

Status: recorded design response to Astra F5, not implemented or GPU-validated.
No code, model workload, training, service change, commit or deployment is
authorized. All future model workloads use the explicitly selected RTX 3090;
no CPU fallback. Follow [specification](architecture-and-training-plan.md),
[B1 role mapping](neural-input-encoding-v1.md),
[resource admission](position-limits-and-budget-accounting.md) and
[losses](loss-configuration-and-training-schedule.md).

Owner: execution states, eligible decisions, native vocabulary masks, writer
step semantics, consumption count, phase closure, notices and terminal outcomes.
This profile intentionally revises the former +32 consumption-position hold to
ZERO, and assigns mechanical code 12=WriterAborted. B1 field widths are unchanged.

## 1. States and decision boundaries

| State | Adapter | Allowed work |
|---|---|---|
| Prefill | D | One prompt prefill; fixed budget readout |
| Startup delivery | R | Three-position budget return |
| Native thinking | R | Operation gate, then one native token if Continue |
| Native answering | R | Operation gate, then one answer token if Continue |
| Writer | R | Initialize arguments; bounded seven-action ticks |
| Evaluation | D | Admitted request; status/value readout |
| Return delivery | R | Entire validated return |
| Closing thinking | R | Fixed closing sequence |
| Mechanical notice | R | One bounded runtime control input |
| Terminal | None | No further model work or delivery |

Evaluate the operation gate once per eligible native boundary. Never gate during
startup/writing/evaluation/return delivery/closure. Continue samples one permitted
native token; it does not poll the same unchanged hidden state again. A latent
call can resume the same phase at its return boundary without a text token first.

At native boundaries: cancellation first; reject unusable/nonfinite execution
state; apply forced phase/answer limits; gate with structural masks; sample a
native token only on Continue. No nested request or second prefill.

## 2. Startup and fixed chat closure

Validate configuration including A>=1 before model work; preserve P+A+H<=C.
Always use the pinned thinking-enabled assistant prefix for a valid job. Its
opening <think> is already part of P. Prefill once under D, read initial budget,
determine actual grant/override, switch R and deliver BUDGET_HEADER/BUDGET_VALUE/
BUDGET_END (three H positions). BUDGET_VALUE remains detached clamped predicted s;
actual grant/override is in counters. If B=0 close immediately, otherwise think.

Static pinned tokenizer entries:

| Text | ID |
|---|---:|
| <think> | 248068 |
| </think> | 248069 |
| Two newline characters | 271 |
| <\|im_end\|> | 248046 |
| <\|endoftext\|> | 248044 |

Canonical closing sequence is IDs [248069,271], decoding to </think> followed
by two newlines. Both spend H, not B/A. Begin closure on model-selected </think>
or B=0 at a native-thinking boundary. Disable gating through the entire sequence;
switch phase to answering only after both positions commit. For natural closure,
commit the already selected closing token ONCE, then the fixed separator.
Precheck the sequence capacity; do not begin an ordinary closure that cannot fit.
Cancellation/device failure can terminate an in-flight sequence without resumption.

## 3. Native sampling and stop masks

Ordinary native vocabulary is base IDs 0..248043. Additionally allow only
</think> in thinking, and <\|im_end\|> in answering. Mask <think>, wrong-phase
controls, other input-only protocol tokens, private scalar rows and unused rows.
Explicitly set chat EOS=248046. Do not inherit model-config EOS=248044 when the
tokenizer uses chat end=248046; <\|endoftext\|> is masked in this profile.

Only exact generated control IDs at eligible native boundaries trigger changes;
do not scan decoded substrings or react to source/historical packet tokens.
The ordinary newline token 271 spends B/A if sampled as text, H only when it is
the fixed closure separator. Control/forced origin is explicit in telemetry.

Thinking: ordinary token spends B; selected </think> begins closure; B=0 forces
closure before another gate/text token. Answering: ordinary token spends A;
selected chat end spends H and completes normally. At A=0, append forced chat
end if safely possible, then terminate Limited without another gate/text token.
Commit every emitted token/control input once on the existing cache. Completion
means a normal generation stop, not semantic correctness; empty answers may fail
semantic evaluation. Model-selected termination is distinct from forced closure.

## 4. Exact writer step semantics

WRITER_START initializes Q/T and inline fields; it executes no seven-action
writer event. Each following WRITER_TICK selects and executes ONE action from
its resulting hidden state. FINISH/ABORT each occupy a real writer position.
Thus ordinary preparation remains start + record steps + finish.

Successful record actions commit validated records. Recoverable malformed
proposals leave the previous draft intact; next tick carries the mechanical
error in its existing metadata. Valid FINISH seals/admission-checks/dispatches.
Structurally invalid FINISH remains unsealed and may repair if another step fits.
ABORT ends without a semantic result. Exhausted writer/B/context/object capacity
ends mechanically. The previous-action field identifies the last attempted
writer action, including unsuccessful execution; successful repair clears the
pending error. Initialization is not an eighth action. Nonfinite/unusable model
state is fatal, not an endless repair exercise.

If position 32 does not successfully FINISH or ABORT, close with WriterLimit;
never append an uncharged 33rd writer step. For simultaneous admission failures,
report the first in fixed order: record capacity, handle capacity, B, context.
Retain other violations in telemetry. Preserve dedicated sealing/completion holds.

## 5. Consume return and resume: zero extra ticks

During evaluation/return delivery, disable native sampling and gating. Validate
the return before publishing/delivery. Deliver the full packet under R, mark
consumed exactly once at RETURN_END, release unused completion reservations and
resume the SAME native phase from that hidden state. Answer-phase calls do not
reopen thinking or replenish B. Late/duplicate deliveries cannot inject feedback.

CONSUME_TICK role 37 is not emitted. Its count AND reservation are zero. Further
ordinary reasoning spends B normally; another call can begin at an eligible
boundary within remaining limits. Processing the return itself is native result
consumption, not a promise that an untrained reader already understands it.

    position_hold = q + return_max

This REPLACES the earlier q+return_max+32 profile. Record/handle holds unchanged.
Minimum Noul hold=13+50=63. Maximum full attempt=32+256+64=352. The six-candidate
255-position boundary example now holds 255+50=305 positions. Future nonzero
consumption ticks require a new explicit control/admission profile, not silent
activation of the former ceiling.

## 6. Bounded mechanical notices

A closed failed/aborted attempt receives at most ONE RUNTIME_LIMIT input under R,
charged to H. It carries only mechanical metadata/originating attempt state,
never a semantic Result/Reason. Code 12=WriterAborted; 13-15 remain reserved.
Resource non-admission uses its actual capacity code; writer limit uses its code;
a bounded invalid return may use its failure code if the cache is usable.

Recoverable errors INSIDE an ongoing writer draft use its next charged B tick,
not repeated H notices. After the closing notice resume the original phase;
B=0 in thinking forces closure before gating. After 16 attempts mask optional
operations; do not repeatedly emit notices because calls are no longer allowed.
H cannot pay for semantic reasoning or become a new unbounded computation loop.

## 7. Terminal roles/outcomes

PHASE_CLOSE role 38 and TERMINATE role 40 are not emitted in v1; native chat
delimiters perform those transitions. IDs remain defined for possible future
profiles. No post-terminal inference is needed to learn that the job ended.

| Outcome | Meaning |
|---|---|
| Completed | Native chat end selected normally |
| Limited | Answer/resource limit stopped job |
| Cancelled | User cancellation |
| Failed | Fatal execution/protocol-state failure |

Release unused holds on every terminal path; preserve spent positions/published
records. Exactly one terminal commit. A later cancellation does not change a
committed terminal outcome. No silent cache restart/rollback or CPU fallback.

## 8. Cancellation and unsafe failure

At cancellation stop scheduling, discard late output/returns and append no
notice/closing/EOS tokens. Retain charges for work already started. Wait for
in-flight GPU work to settle before releasing model-operation ownership. Stale/
duplicate completions are dropped without new model positions or cache mutation.

A bounded malformed return can receive one notice only if causal state remains
demonstrably usable. Nonfinite state, unsafe partial mutation or device failure
ends Failed out of band. If required controls cannot fit, terminate out of band;
do not partially continue, substitute values, replay input or switch to CPU.

## 9. Control reserve bound and training labels

| Work | Maximum H positions |
|---|---:|
| Startup | 3 |
| One thinking close + separator | 2 |
| One notice per attempt | 16 |
| One normal OR forced chat end | 1 |
| Total | 22 |

H=64 is retained, leaving 42 positions of margin in a healthy profile-conforming
run. The bound assumes the finite notice rule, one phase closure, no extra ticks
and no post-terminal work. It does not establish GPU execution/semantic success.

Log model-selected versus forced controls, token origin, debits and stop reasons.
Forced formatting/limit tokens are not native answer-token targets. Model-selected
closing/end tokens can receive the annotated native termination supervision.
Decision forcing and native text teacher forcing remain separate dimensions.
Required future fixtures include natural/forced close, B=0 startup, answer-phase
calls, malformed writer repair, caps, aborted/not-admitted attempts, EOS config
override, duplicate/late completion, cancellation races and unusable-state exits.
