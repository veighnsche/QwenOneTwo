# Episode producer and latent invocation proof

Implemented against the existing training release. No new scenarios, model-service
calls, agents, reference fitting or optimizer updates. The Operating Table and
live services were unchanged. [Portable receipt](evidence/episode-producer-v1.json)
binds source files, local logs, release hashes, fixtures and measured outputs.

## Slice 1: execute the packet boundary

Acceptance: an optional Qev invocation rejects text/JSON, runs encoded FP32 CUDA
packet vectors through W+D in the existing cache, and consumes its bound return
through W+R. Source states remain connected to the original prefill. Startup
retains its continuous scalar, existing ordered S-curve and distinct allowance.

`Session.start_budget` allows the producer to register connected source states
between its sole prefill and startup. `Session.evaluate` explicitly requires a
`Packet`, rerenders its authoritative snapshot and resets evaluator capture even
on failure. `CleanPackets.request(session=...)` executes its two real preparation
positions in that session instead of treating offline ledger charges as forwards.
Supplied reader returns do not charge an already evaluated request a second time.

**Textless concerns optional Noul/Choice/Score invocation.** The original human
prompt is still text and its initial W+D prefill still reads effort, as required.
English role annotations may be encoded by the separate offline frozen reference;
they never become a second textual live Qev prompt.

## Slice 2: materialize and execute existing records

Acceptance: immutable selected release records produce complete sessions and
explicit loss/exposure metadata, or honest failed exposures without prefix losses.
Current outcome annotations cannot enter writer/request encoders. Gold returns
are confined to isolated reading. All model execution stays on the named RTX 3090.

- `training/producer.py`: `EpisodeProducer` is the native `Driver` callback.
  It takes the backbone, codec and `load_release(...)` result. Each call receives
  `(release.record(id), phase, window)` and returns `(Session, losses, Exposure)`.
  It never calls an optimizer or launches a run. Native source memory uses pinned
  tokenizer byte offsets and connected states from the actual prefill. Guided
  operations/structure use live semantic emissions and actual Qev returns. Later
  guidance stops after an unverified return; transition/free controls follow A4.
  Direct/language targets stay separate from prompt inputs. Early teacher forcing
  is explicitly declared; transition/free execution uses the production policy.
- `training/component_producer.py`: clean receiver, isolated reader and decision
  replay materialization. Supply `reference=`, `checkpoint_hash=` and
  `profile_hash=` to `EpisodeProducer` for these branches. A missing/nonfrozen/
  changed reference rejects. Byte-bound permitted facts and per-role glosses
  construct packets through `CleanEncoder`/`CleanPackets`. Reader prefixes omit
  original evidence; nonselected alternatives retain only binding identities,
  with selected contents and justified non-assessed Reason separately permitted.
  Reason status remains outside request roles. `reference_record` and
  `reference_record_objective` select existing reference-stage annotations;
  the latter returns the existing objective graph and never steps an optimizer.
- Overlength guards run before debiting unexecuted positions. Failed/overlength
  episodes return no losses, preserve ledger/cache equality and settle holds.
  Isolated decision replay remains distinct from forced native decisions in a
  free-control window; forced joint/native policy still rejects.

Training/development/locked manifest hashes remain exactly those in the handoff.
No release files or historical evidence were rewritten. Locked scenarios were
not used to select or repair fixtures. Model-free boundary tests use visibly
synthetic schema fixtures; GPU tests load actual existing training records.

## Measured evidence

**57/57 model-free tests passed in 2.217 s**:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_episode_producer test_training_inputs test_training_driver test_training_release test_training_schedule test_data_production -q
```

**11 distinct focused GPU tests passed**, using only
`CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1` and the pinned base
snapshot/Qev/diagnostic artifacts. Logs are in `data/episode-producer-20261105/`.
Eight actual tests passed in the 77.136 s primary run. That command also contained
the incorrect selector `test_codec_gpu.CodecGPU`, so its overall exit was 1;
the loader error is retained, not presented as a passing command. The new
free-window guard and corrected
`test_codec_gpu.FullCodecGPU.test_zero_budget_factory_and_direct_native_helper_preserve_one_cache`
then passed **2/2 in 15.833 s**, without repeating the eight model checks.
A final tied-choice reader regression passed **1/1 in 16.793 s**: set-valued
selection annotations do not supervise a single selected return. Earlier clean
3/3 (42.600 s) and 7/7 (65.318 s) runs are also retained.

The decisive transport test disables **both text embedding lookup and text
tokenization throughout construction, invocation and delivery**. It verifies
that the exact `codec.encode(Packet)` tensor enters the single D request forward,
followed by R delivery. A native-token loss after the actual return has finite,
nonzero derivatives into that same tensor. All three sessions terminate with one
prefill, lineage 1, no held positions and exact P+B+A+H accounting:

| Existing record | Port | Request tensor | P / B / A / H | Total | Request gradient norm |
|---|---|---|---|---|---|
| train-writer-0004 | Noul | 103 × 1024 FP32 CUDA | 97 / 126 / 1 / 6 | 230 | 148.95587158203125 |
| train-writer-0000 | Choice | 143 × 1024 FP32 CUDA | 121 / 170 / 1 / 6 | 298 | 935.7823486328125 |
| train-writer-0001 | Score | 110 × 1024 FP32 CUDA | 93 / 141 / 1 / 6 | 241 | 2565.18359375 |

Each dispatch and consumption count is 1; textual invocation token count is 0.
Qev predicted Noul Insufficient and Choice/Score Assessed. No expected status or
value was substituted. Tests separately reject strings, bytes and dictionaries
before a model forward. Actual clean receiver/reader/replay records completed at
169/74/169 positions; non-assessed Noul/Choice readers at 57/73 positions.

## Limits and next unfinished slice

Positive transport fixtures prescribe operation/writer structure and **Reason
attachment count zero through a test-only head-forward override**. They leave
Qev status/value/selection and live semantic emissions actual; no parameters are
updated. Clean component tests freeze a freshly instantiated **untrained**
reference. Fixed `Done` continuation proves derivatives, not answer competence.
None of these fixtures establishes learned invocation or semantic correctness.

The unmodified diagnostic Reason decoder actually fails by selecting a historical
result when no compatible result exists. A regression proves that the producer
records Failed, one dispatch, zero consumption, zero losses and released holds,
with no gold repair. The original failed run remains in `gpu-initial.log`.

**Next: qualify the frozen reference and bind actual predicted request semantics
to reviewed supervision.** Native joint semantic/outcome and post-return
continuation losses remain masked until that binding exists. Writer alignment,
semantic reconstruction/contrast targets, compatible multi-call continuation
binding and observed correction materialization remain unfinished integrations;
prescribed correction specifications do not supply observed failures. Generic
per-candidate `unknown` feasibility is masked rather than inventing whether it
means Insufficient or Conflicting. Set-valued reader annotations also mask
native continuation loss until bound to a reviewed single selected return.
Full participating trainer-memory qualification
also remains required. No training is authorized by this implementation.

The 324-case and 108-case numeric matrices were not rerun. Qualified v3 arithmetic
is unchanged; original BF16 transformer arithmetic remains failed. Passing these
producer slices does not certify the complete learned flow or promotion.
