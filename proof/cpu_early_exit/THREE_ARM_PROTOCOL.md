# Matched CPU systems experiment

The fair synthesis comparison has three arms on identical decoded latents,
frame geometry, QPs, precision, thread count, NUMA placement and output
boundary:

1. **Released D12, optimized FP32 backend.** Apply the same oneDNN/ORT
   compilation and legal fusions used in arm 2 to Microsoft's released
   decoder. This is the systems baseline.
2. **RegLIC e15, same optimized FP32 backend.** Include the decoder-side
   router, shared stem, tile scheduling, variable-depth suffix, adapters,
   seam repair, assembly and head. Its difference from arm 1 is the
   architecture and routing, not a weaker baseline implementation.
3. **RegLIC e15, same backend plus measured custom fusion.** Replace only
   the profiler-selected kernel(s), verify output equivalence and measure
   the incremental gain against arm 2. A hand-written kernel is not assumed
   beneficial merely because its microbenchmark improves.

The current `matched_routed_cpu.py` implements the **unoptimized PyTorch FP32
control** for arms 1 and 2, plus an all-deep e15 control that pays the same
router/stem overhead. It does not implement or name a custom-fusion arm.
`bench.py` verifies FP32 ONNX export for released D12 and *uniform*, full-frame
e15 exits. Those exports do not represent routed e15. A fair ORT/oneDNN arm
requires partitioning the e15 execution into compiled stem, per-depth suffix,
adapter, seam-repair and head graphs while retaining the same host scheduler
and routing decisions. The router's own cost stays inside e15 latency.

The pinned e15 checkpoint uses `split_depth=2`, six exits, a 256×256 RGB tile,
replicate tile padding and grid seam repair. Any compiled suffix must preserve
these details. Kodak 01 QP16 supplies a nontrivial route (five tiles at exit 3,
one at exit 4); Kodak 01 QP32 routes all six tiles to the deepest exit under
the current fixed-QP beta. Both controls are needed: QP32 alone cannot show
the benefit of early termination. The same decoded latent was used in every
arm, with all 255 nondecoder checkpoint tensors equal. The QP16 smoke gives
released/routed/all-deep YUV611 4:4:4 PSNR of 29.589/29.504/29.550 dB.
These are quality checks, not latency results. The research FUFREF2 stream is
not Microsoft's native CUDA wire format.

For the AVX2 kernel, the exact operator matters. Microsoft's activation is
`x * sigmoid(4*x)`, not ordinary `silu(x) = x * sigmoid(x)`. Its
`WSiLUChunkAdd` then sums four interleaved output-channel groups. The useful
fusion target is therefore a blocked 1×1 convolution whose 4C output stays
in registers/cache while bias, `sigmoid(4*x)` and the four-way reduction are
applied. OneDNN post-ops may fuse the elementwise activation and sum, but the
interleaved channel reduction needs separate validation or a custom C++
kernel. Preserve the original arithmetic order where possible and test the
final YUV PSNR across images and QPs; an isolated small tensor error is not
enough. AVX2/FMA is the relevant ISA on EPYC 7443; AVX-512 is unavailable.

Claim gate: an idle host, a pinned oneDNN EP build whose provider is verified,
randomized paired block timing on at least Kodak 3×3 streams, 1/2/4/8 threads
with node-local affinity, complete raw samples, output equivalence and YUV
quality. End-to-end bitstream-to-image timing is reported separately from
synthesis timing. Until those checks pass, no CPU speedup factor belongs in
the paper. Likewise, the earlier GPU ~2.76× figure is a synthesis-only
comparison and cannot be labelled a full-codec 2.81× result.

The ONNX Runtime oneDNN EP needs a wheel built with `--use_dnnl`; the stock
wheel's `CPUExecutionProvider` uses a different backend. See the
[official build instructions](https://onnxruntime.ai/docs/build/eps.html),
[oneDNN EP guide](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html)
and [oneDNN post-op API](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html).
