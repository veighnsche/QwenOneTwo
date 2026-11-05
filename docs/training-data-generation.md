# How the training data will be made

Status: recorded production procedure, not an existing generated/reviewed
dataset. No coding, model workloads or deployment are authorized.

Owner: scenario production. Use the
[data/readiness contract](training-data-and-readiness.md) for labels and splits,
and [handoff](implementation-handoff.md) for the feasibility-first work order.
Dataset serialization may use ordinary inspectable records; the textless
constraint applies to deployed internal calls, not the on-disk annotation format.

## Structured situations before wording

Build fictional situations with inspectable facts, supplied rules and
independently checkable outcomes. For example: workshop access requires completed
orientation and an active badge; Sam completed orientation but his badge expired.
This is sufficient to label the access proposition false without asking Qev to
invent its own truth labels.

Produce ordinary human requests from the situation. Do not mention Noul, Choice,
Score, latent calls or routing in those requests. Store the intended question,
evidence bindings, candidate bank/rubric, semantic status, decision reference
and appropriate continuation separately for training and review.

## Controlled variations and splits

Vary substance as well as wording: changed eligibility, omitted evidence,
contradictory sources, different questions over the same facts, several
alternatives, changed score anchors, and valid direct-answer tasks. Split
scenario families before generating variants; keep closely related versions
together. Exclude the saved natural prompts and their close paraphrases from
training/tuning.

Start with authored templates. A GPU-hosted teacher may later vary wording and
propose annotations, but verify fact preservation, missing-information handling
and absence of operation hints. The teacher is not the source of truth. Teacher
selection and actual generation remain future work, not an existing capability
or authorization to start a model service.

Deterministic scenario construction and bookkeeping may use ordinary scripts.
Any model generation, annotation or encoding must use the selected GPU.

## Training trajectories

An inspectable trajectory records the human prompt, an acceptable operation and
invocation window, intended question and arguments, reference status/result,
appropriate continuation and final answer. Mark guided actions and supplied
gold returns explicitly as supervised. Do not present an annotated desired
trajectory as a run the model performed.

Early training may guide actions and supply correct returns to isolate skills.
Later whole-flow training uses Qwen's actual latent requests and Qev's actual
results, with reference annotations supplying losses. In free evaluation no
correct request, forced action or gold return is injected.

Do not manually assign 1024-dimensional vectors. Annotate semantic meaning and
exact data. Train the latent writer, decision receiver and native reader to
communicate it, following the recorded representation and gradient strategy.
Offline English annotations do not become inference-time tool-question text.

The [semantic-supervision method](semantic-supervision.md) whitelists per-record
inputs to an answer-blind reference encoder. Reference decisions and desired
scores never enter its question/rubric inputs. Freeze the semantic target basis
before writer/Qev joint training; factual candidate attributes remain legitimate.

Use the complete [training-input whitelist](training-input-rules-v1.md), including
neutral Target, value-free Attribute-key and permitted post-decision Reason
annotations. Reason status/support is explicitly allowed only in that role;
source facts may bear relevant outcomes with provenance. Component exercises,
guided prefixes and actual joint returns retain distinct exposure labels.

## Pilot before scaling

The [data-operations owner](data-operations.md) fixes the initial pilot at 120
authored scenarios, 20 per domain, with deterministic templates and no model
generator dependency. It owns independent review/adjudication, provenance and
release rules. Do not describe model-assisted proposals as reviewed gold.

Review about 120 distinct scenarios across the six domains, then controlled
variants. Check deterministic labels against scenario rules. Review subjective
rubrics and acceptable intervals; revise unclear cases. Record generation,
checking and review provenance. These scenario/annotation reviews precede
spending GPU time on the full training set.

See [scope, annotation and readiness](training-data-and-readiness.md) for initial
quantities, review policy, split design and proposed acceptance gates. The
[complete worked examples](complete-training-examples.md) are review drafts,
not generated latent vectors or training results.
