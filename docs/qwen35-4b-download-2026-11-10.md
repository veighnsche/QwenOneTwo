# Qwen3.5-4B and decision LoRA — 10 November 2026

The requested weights are ready on disk. This was a download and static
verification task; no inference, fitting, training or runtime migration occurred.

| Artifact | Pinned revision | Local files |
|---|---|---:|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9,342,905,899 bytes, 13 files |
| [evalengine/decision-4b](https://huggingface.co/evalengine/decision-4b) | `aadc55517ae8bfb1264fd5932ec452b939361e47` | 65,042,061 bytes, 11 files |

The two base weight shards were already cached; missing README/license files
were completed. The decision adapter and accompanying documentation/license/
benchmark files were downloaded. No model or downloaded repository code ran.

## Selection

Decision-4B was selected as a strong compatible candidate: its adapter explicitly
pins the exact Qwen instruction-tuned revision above. Its author's same-panel
comparison reports 79.1% accuracy and 76.4% family mean across 2,800 cases, with
one forward pass. The panel covers familiar training-source distributions;
this is not independent proof of universal superiority or our task performance.
[Source](https://huggingface.co/evalengine/decision-4b#benchmark).

Other inspected options included
[Kahn1 LoRA](https://huggingface.co/Okura66/Kahn1-Qwen3.5-4B-LoRA), whose reported
benchmarks use three option orders/calibration;
[AustinFu Qev-4B](https://huggingface.co/AustinFu/Qev-4B), which targets the
different `Qwen3.5-4B-Base` checkpoint;
[rycerzes Decision](https://huggingface.co/rycerzes/Qwen3.5-4B-Decision), published
as merged weights; and
[wozniakos10 typed-decisions](https://huggingface.co/wozniakos10/Qwen3.5-4B-typed-decisions),
which declares an Unsloth base and additional decision head. Their reported
benchmarks are not one interchangeable ranking. Only Decision-4B weights were
selected/downloaded; small adapter configuration files were inspected as metadata.

Decision-4B's published interface uses textual state/question/options with option
logits, and its documented context limit is 2,048 tokens. It is not a ready-made
replacement for the project's latent protocol, continuous task-Score row or
trained 0.8B artifacts. No integration or acceptance change was made.
[Interface and limitations](https://huggingface.co/evalengine/decision-4b#try-it).

## Verification and paths

- Base snapshot:
  `/home/vince/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Adapter snapshot:
  `/home/vince/.cache/huggingface/hub/models--evalengine--decision-4b/snapshots/aadc55517ae8bfb1264fd5932ec452b939361e47`
- [Portable receipt](evidence/qwen35-4b-download-2026-11-10.json) records each
  file's size/hash and upstream integrity verification, revisions and snapshots.
- Large files match upstream LFS SHA256; other files match upstream Git blob
  identities and have local SHA256 receipts. Both base shards were hashed.
- Adapter rank 8 / alpha 16; 496 adapter tensors form 248 matching LoRA pairs.
  Shapes were checked against 738 base tensor headers, without reading tensors
  into a model runtime. The text-only `model.layers.` names map to the full
  checkpoint's `model.language_model.layers.` names. Initial strict-name rejection
  is preserved in the raw log; the explicit mapping resolved that metadata check.
- PyTorch, Transformers, PEFT, Safetensors, NumPy and CuPy imports were blocked
  during download/verification. Static shape agreement is not a runtime PEFT
  load, numerical, inference-quality or full-context memory qualification.
- Raw metadata, verifier and logs: `data/model-downloads-20261110/`.

Post-download device inspection found no GPU compute processes: RTX 3090 used
29 MiB, RTX 3060 11 MiB. `qwen-one-two.service` remained inactive, MainPID 0,
following the user's preceding unload request. All weights remain unloaded.
Existing 0.8B checkpoints, code, model selection and live services were not changed
by this download. RD-09 remains unstarted.
