# DCVC-UF depth codec: causal CPU reference

This experiment provides actual coded bytes and an independently executable
decoder for the ongoing D2/D4/D6 depth study. It is a **CPU FP32 research
format**, not the released CUDA wire format and not a deployment benchmark.
The D12 anchor uses the released checkpoint; shallow checkpoints are frozen
at epoch 20 of 105. These are interim validation results.

## What is checked

The decoder receives the byte stream and its resident checkpoint. It first
decodes the hyperlatent, then derives the Gaussian indexes for each of the
four spatial-prior stages from information already decoded. No encoder-side
indexes, source image or analysis features are passed to the decoder.
The isolated worker disables the encoder, hyper-encoder and training-forward
entry points so that accidentally invoking them fails immediately.

The pinned upstream rANS kernel is built in a separate directory. The only
binding change exposes an owned NumPy copy of decoded symbols. The training
environment, frozen upstream checkout and live trainers are not modified.
Because rANS is a stack, encode submissions are `y3,y2,y1,y0,z`; decode
consumes `z,y0,y1,y2,y3`.

Quantization follows the deterministic FP32 training-forward convention:
rounded symbols, no threshold skipping, and no silent int8 clipping. A symbol
outside [-128,127] is an explicit failure. The released CUDA implementation
uses a different numerical path; compatibility is not assumed.

The `FUFREF1` container has an 88-byte header: dimensions, depth, QP, numeric
format, payload size, checkpoint SHA-256 and payload SHA-256. Sixty-four header
bytes are research integrity/identity hashes. Report **payload bytes**,
**whole-container bytes** and **estimated entropy** separately. The header is
not a claim about the minimum signalling cost of a production codec.

## Verification and validation

`verify_reference.py` checks D2/D4/D6 and released D12 at QP 0/32/63 on
64×64, 65×97 and 512×512 inputs. All 36 cases passed on 27 September 2026:
fresh-process reconstruction, latent hashes and coder-index hashes match
exactly, and the reconstruction equals the official FP32 model forward.
Malformed containers and unsupported symbol values are rejected.

The check caught a numerical-layout issue: a 1×1 hyperlatent's permuted tensor
can be considered contiguous while retaining a different channel stride.
That layout selected a different CPU convolution path. An explicit
`clone(memory_format=torch.contiguous_format)` in the decoder restored exact
agreement. This engineering finding does not establish cross-device
determinism.

`prepare_validation.py` creates hash-verified centre512 crops of all 100
DIV2K validation images (0801–0900), without resizing. `evaluate_validation.py`
evaluates each frozen model at QP 0/16/32/48/63: 2,000 encode/decode cases.
The decoder is a separate persistent process per model. Every case retains
its actual stream, measurements, source identities and equality checks.
The first four training-monitor crops remain explicitly identifiable;
the additional 96 are reported separately where relevant.

The checkpoint epoch and evaluation settings are fixed before the run; this
validation does not choose checkpoints. CPU diagnostic times include Python,
hashing and correctness work and must not be quoted as GPU codec latency.

## Reproduce on this server

The pinned upstream revision is
`cbdae87a5445114cdc7f48816da63ea80bdeac40`. The experiment root is
`/data10/shareddata/can_karsal/dcvcuf_depth_20260927`; `model_io.py` records these
paths explicitly. The interpreter is `$RUN_ROOT/venv/bin/python`.

```bash
RUN_ROOT=/data10/shareddata/can_karsal/dcvcuf_depth_20260927
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
"$RUN_ROOT/venv/bin/python" experiments/dcvcuf_reference_20260927/build_entropy.py
"$RUN_ROOT/venv/bin/python" experiments/dcvcuf_reference_20260927/verify_reference.py
"$RUN_ROOT/venv/bin/python" experiments/dcvcuf_reference_20260927/prepare_validation.py
nice -n 10 "$RUN_ROOT/venv/bin/python" experiments/dcvcuf_reference_20260927/evaluate_validation.py
```

Run these sequentially, after any active evaluation has completed. The
evaluation pins its code and manifests; changing them mid-run invalidates its
provenance. Resume uses matching existing case files and checks stream hashes.
Check `research/div2k100_reference_epoch020/progress.json` for completion or a
failure. A partial run is never treated as a complete benchmark.

The independent depth bank still needs an optimized depth-aware GPU path,
matched D12-scratch training, fixed-depth patching controls, and measured
selector/grouping/entropy costs before an adaptive runtime claim is supported.
