# Native depth support: prepared patch, not a validated GPU implementation

The pinned upstream CUDA image decoder explicitly declares and invokes twelve
trunk blocks. Truncating the Python `dec.dec_1` alone does not make the native
`compress`/`decompress` path depth-aware. Four operations must agree:
parameter loading, activation preallocation, forward execution and pool release.

`prepare_patch.py` produces a reviewable patch for those operations. It replaces
twelve individual block members with a vector and infers the retained prefix
from canonical state-dictionary keys. Only 2/4/6/8/10/12 contiguous blocks are
accepted. The upsampler and output head remain. A proxy can be initialized once;
changing checkpoints requires a fresh instance so captured CUDA graphs cannot
silently retain old parameter/buffer pointers. Coding before initialization and
batch sizes other than one are rejected.

The patch is **not applied to the training checkout**, installed in the active
environment, compiled as a CUDA extension, or used for reported measurements.
The independently compiled, CUDA-free key parser accepts six valid depths and
rejects 58 malformed states. `git apply --check` passes on the pinned clean
revision. Those checks do not validate CUDA compilation or numerical parity.

```bash
python3 experiments/dcvcuf_native_depth_20260927/prepare_patch.py
```

Prepared source files and the helper executable are under
`/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/native_depth_draft`.
`native_depth_draft.patch` is the portable change, and
`preparation_report.json` records source hashes and remaining GPU checks.
The upstream revision is `cbdae87a5445114cdc7f48816da63ea80bdeac40`.

## Isolated compilation attempt

`build_isolated.py` was started on 27 September at16:57UTC. It archives the
pinned source into a separate build directory, applies the draft there,
and uses explicit SM86 flags instead of querying a GPU. CUDA devices are
hidden; one compiler job and one NVCC split-compile thread run at nice19.
Nothing is installed into the training environment.

The dependency is Microsoft's specified CUTLASS v4.4.1, resolved to
`4370102f9dacab813282e1d67722fceb0b90a019`. The local compiler is CUDA12.1
while PyTorch is built for CUDA12.6; that minor-version difference is
recorded in the build manifest. CUTLASS lists CUDA12.x compatibility in its
[versioned requirements](https://github.com/NVIDIA/cutlass/blob/v4.4.1/README.md#compatibility).
Compilation alone cannot establish that this particular extension is
correct under that toolchain.

Inspect `research/native_depth_build_sm86/status.json` under the experiment
root for completion; a final `compile_report.json` is written here when
the attempt ends. A successful build still needs every GPU check below.

## Required validation when training GPUs are available

1. Build in a separate inference environment. The upstream build script
   queries CUDA device zero and compiles many CUTLASS kernels; do not invoke it
   while treating this as a CPU-only study.
2. Compare stock and patched D12 on the same GPU, QPs and input geometries.
   Require matching coded payloads and reconstructions before using the
   patched implementation as the depth-study reference.
3. Strictly load each depth's checkpoint in the Python architecture first;
   confirm declared depth against retained key count. A missing whole block
   pair can otherwise resemble a valid shallower prefix.
4. Verify actual D2/D4/D6 execution, then D8/D10 when those weights exist.
   Test multiple shapes/QPs and graph recapture. Do not substitute padded
   D12 weights or no-op blocks as a runtime result for a shallow network.
5. Decode in a fresh process after explicit proxy initialization, without
   calling `compress` on the source image. The original Python wrapper's
   lazy proxy initialization lives in `compress`; decoder-only setup must
   create the proxy and install model/CDF parameters directly.
6. Test serial switching between resident experts and output lifetimes.
   The upstream tensor pool is global; concurrent expert streams are a
   separate buffer-ownership problem, not implied by this patch.
7. Measure actual bytes and synchronized wall time in the same inference
   mode for all models. Report initialization/warm-up separately.

## Scope discoveries from the native source

The released `compress` call reconstructs an image on the GPU while its
entropy worker performs symbol transfer and rANS coding. Those branches can
overlap. Summing a separately timed neural decoder and CPU entropy duration
does not give encoder wall time. The intra-image use case could investigate
omitting encoder reconstruction, but that is a separate execution-contract
ablation and is not implemented here.

Native allocation also hardcodes batch one. A model-bank cost expression
using `tau_k(n_k)` is prospective: real batching of multiple patch streams
needs additional implementation and correctness checks. Serial patch calls
must not be labelled a batched native codec benchmark.
