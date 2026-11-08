# Direct role states v1 — opt-in layout and admission contract

This RD-04 profile is experimental and **not GPU-qualified**. It adds no training
authorization or default-profile conversion. The existing
`b1-full-v1-compensated` layout remains the fixed-slot control.

`direct-role-states-v1` carries ordered content-token states of each complete
role-whitelisted canonical annotation. Chat-template positions are excluded;
the complete template/configuration and content mapping are fingerprinted. No
pooling, bins, truncation or semantic annotation text enters a live request.

Question, target, each alternative, each rubric subfield and each attribute key
have explicit lengths **1..32**, width 1024, FP32. The request role order remains
question, target/attributes, evidence, rubric, ordered candidates, evaluate.
Rubrics declare five lengths in dimension/conditions/zero/one/midpoint order;
absent midpoint has length zero and emits no padding vectors. Records and packets
carry the profile explicitly. Mixed profiles reject; references are never
silently converted. Registry ownership clones tensors without detaching graphs.

The shared live role projection is `x + linear(x, delta_weight, delta_bias)`,
FP32 with exactly zero initial delta. It runs before the existing normalization
and wire framing for the above roles only. Raw role states remain immutable;
the projection is a live consumer with one named owner. A shared, role-conditioned
length head has 33 logits; zero is masked for present fields. Native emissions
remain parallel per writer event, with capacity 32 vectors per field. The writer
still has 32 events, 32 allocated records and 16 attempts; no allowance increases.

Exact typed atoms, values, source spans, references, units and provenance retain
their ABI. Variable key lengths affect attribute position costs, not values.
Return Reason remains the existing four-slot control plus at most 32 attachment
positions; it is not a direct annotation input or a new semantic qualification.
Supplied-reader annotation support is separate work, never silently pooled into
a direct request. Return echoes include complete question/target/rubric vectors,
and assessed Choice returns the complete selected candidate and attributes.

Metadata version **2** identifies this profile. Header reserved bits 57..63
carry question length; target/candidate/attribute descriptor bits 57..63 carry
their own field/key length. Rubric descriptor bits 96..125 carry five six-bit
lengths, with 126..127 zero. Other applicable fields and exact atom encodings
retain their positions. Legacy metadata version 1 remains byte-for-byte unchanged.
All lengths also live in immutable typed records; headers and descriptors must
agree with rendered role ordering and actual vector counts. Candidate ends follow
the complete candidate, including attributes.

Admission is defined in `codec/layout.py`. With `E = 1 + Q + 1 + T + rubric_cost`,
request cost is `E + target_attributes + evidence + candidates + 1`; an assessed
return is `E + scalar_or_selected_candidate + 1`; worst-case abstention return is
`E + 1 + 4 + 32 + 1`. Every candidate's full return cost is considered, independent
of predicted selection/status. The bounds remain **256 request /64 return**, and
source/history spans and handle/context/budget/record reservations remain intact.
The old 50/61 reservations are used only for the legacy layout.

Examples without attributes/evidence: Q8/T2 Noul costs request 13 and worst-case
return 50; Q8/T2 plus four rubric fields of length 2 costs Score request 22 and
return 59 (no absent-midpoint padding); Q16/T12 Noul rejects because its maximum
return is 68. Field length 33 rejects before execution. Coverage loss must list
every rejected row and operation/family; no model-performance selection or hidden
common-support subset is permitted.

`RD04-GPU-IDENTITY-PROJECTION`, `RD04-GPU-DIRECT-ROUNDTRIP` and
`RD04-GPU-DIRECT-GRADIENTS` are owned by RD-07 and remain pending. Required static
checks precede staged delivery. None of these definitions certifies learned
invocation, Reason meaning, numeric accuracy or useful behavior.
