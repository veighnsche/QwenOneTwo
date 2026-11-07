# Simple direct-readout diagnostic

Authorized by the user: fit a simple direct readout of saved latent features on
the RTX 3090, frozen W/reference, training/development separate, no agents or new
data, record results and commit/push. This is an offline representation probe,
not a production receiver, learned-invocation test or capability promotion.

Reuse parent-0188 frozen h from the complete 1,003-record training cache and the
saved slice-2 X captures. Train only on reviewed Assessed scalar records:
**256 Noul /254 Score**, 96 training families per operation. Development features
exist for **180 X cases**, including **70 assessed Noul /40 assessed Score**.
The E h tensors were not saved; do not re-extract them or imply all 310 prior cases
are tested here. Choice and non-assessed cases have no scalar target in this probe.
No W/reference or existing codec module needs to be loaded or mutated.

One affine readout per operation directly uses all 1,024 h coordinates with an
unpenalized intercept. It bypasses the fixed prototypes, q/k projections, Score
row and residual bottleneck. This is deliberately a new diagnostic profile;
it does not redefine the production Noul/Score contracts. Fit 0/1 Noul targets
by squared error (linear probability model); Score uses its exact reviewed point
targets represented in the existing FP32 scalar ABI. Reject interval training
targets instead of inventing midpoints. Clamp predictions to [0,1] for diagnostic
metrics; retain raw error/ranges and clipping counts. Noul threshold remains 0.5
with ties failing; Score exact intervals have no added tolerance.

Use a CUDA FP64 dual ridge solve, avoiding an iterative optimizer/convergence
confound. For each training subset, compute mean h and one scalar RMS vector
norm; z=(h-mean)/scale. Minimize mean squared error + lambda*||w||², with
unpenalized intercept. All centering/scaling uses only the fitted training subset.
For each operation choose lambda from **1e-9, 1e-6, 1e-3, 1** using three-fold
family-grouped cross-validation **within training only**. Folds are a fixed hash
of family; no family crosses folds. Exact permitted inputs are unique in these
training subsets. Select lowest pooled out-of-fold **raw** MSE; exact ties prefer
larger lambda. Fit once on the full training subset with the chosen lambda.
This is 24 small CV solves plus two final solves, not a transformer-training sweep.
No development-based selection, calibration, retries, extension or time cutoff.

Export both final probes and their training-only selection evidence **before**
loading development tensors or predicting development values. Fit entrypoints
reject development/locked releases. Pin feature files, receipt/row/input hashes,
parent/reference identities, archived representation source and dataset manifests.
Reject train/development family or permitted-input overlap. Case IDs, labels,
family, folds, text, operation-specific targets and gold returns are never appended
to h. Operation chooses which separately fitted probe is used.

Report training, out-of-fold and development raw/FP32-clipped metrics alongside
the saved parent, joint and value-only scalar predictions on the **same X subset**.
Show Noul true/false coverage, balanced accuracy, Brier and AUC; Score exact hits,
distance and scale direction; and value-sensitive intervention-pair direction and
two-sided correctness. These are scalar diagnostics with reviewed eligibility,
not actual status prediction or delivered behavior. Existing transport/Choice
receipts remain historical evidence; this run executes no new Session.

Acceptance: complete declared fits and held-out predictions, exact provenance
and split separation, finite CUDA-only solves with relative equation residual
<=1e-8, analytic/primal-versus-dual GPU checks, exact export/reload predictions,
unchanged feature/parent files, independent receipt metric/CV-selection audit.
Production forward/backward code is unchanged; reuse its existing qualification.
Probe arithmetic is FP64 with FP32 scalar reporting, explicitly separate from v3.
Record all failures without a CPU fallback or automatic alternative fit.

Interpretation: useful held-out decoding is evidence that this saved representation
contains exploitable information despite earlier receiver failures. A negative
linear probe does not establish that information is absent, that nonlinear decoding
cannot succeed, or that the whole project is impossible. Training success alone
can reflect an underdetermined 1,024-feature fit; grouped validation and held-out
interventions are essential. The frozen reference remains semantically unqualified.
