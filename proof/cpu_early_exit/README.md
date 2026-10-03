# CPU synthesis baseline for DCVC-UF

This is an isolated CPU experiment. It reads a pinned FUFREF2 bitstream,
entropy-decodes the latent on CPU, and exports the **synthesis decoder only**.
PyTorch and ONNX Runtime receive the same latent and QP scale. Y, Cb and Cr
PSNR are measured against the source image after putting the model's centered
YCbCr output back on the 0–1 scale. Latency excludes entropy decode, image
loading, disk I/O and model loading; it is not a full codec speedup.

The [three-arm protocol](THREE_ARM_PROTOCOL.md) locks the released-versus-
RegLIC comparison to the same optimized backend and isolates any future
custom-fusion gain. `matched_routed_cpu.py` is the CPU correctness control
that includes the actual router and tile assembly. Run it with `--blocks 0`
while training occupies the host; nonzero timing blocks refuse a loaded CPU.

`released` is Microsoft's released D12 decoder. `e15_exit0` and `e15_deep`
export fixed uniform exits of the early-exit checkpoint. These do **not**
include decoder-side routing, per-tile variable depth, patch assembly or seam
repair. A routed CPU deployment requires separately compiled stem, suffix,
adapter and head subgraphs plus the unchanged host tile scheduler. Until that
exists, the fixed-exit data must not be called routed performance.

The normal `onnxruntime` CPU wheel uses `CPUExecutionProvider` (MLAS). A
oneDNN Execution Provider measurement requires a wheel built with
`--use_dnnl` and the provider must report `DnnlExecutionProvider`. The script
refuses a silent provider fallback. PyTorch's CPU convolution backend reports
oneDNN availability separately; that does not establish that every operator
used it. Early diagnostic JSON used the provisional label `torch_mkldnn`;
the correct interpretation is plain PyTorch CPU FP32, and later code calls it
`torch_cpu_fp32`. The source for an official DNNL build is ONNX Runtime's
build guide.

For a fresh proof-branch clone, first run the parent
[`bootstrap.py`](../early_exit_vs_released/bootstrap.py) to check artifacts and
build its CPU rANS extension. Install `onnx==1.19.1` and
`onnxruntime==1.24.2` into a separate environment; do not modify the live
training environment. Then run:

```bash
python proof/cpu_early_exit/bench.py export \
  --upstream proof/early_exit_vs_released/.local/DCVC \
  --extension proof/early_exit_vs_released/.local/entropy \
  --model /tmp/cpu/released_d12.onnx
python proof/cpu_early_exit/bench.py quantize \
  --upstream proof/early_exit_vs_released/.local/DCVC \
  --extension proof/early_exit_vs_released/.local/entropy \
  --model /tmp/cpu/released_d12.onnx \
  --quant-model /tmp/cpu/released_d12_int8.onnx \
  --calibration-stream proof/early_exit_vs_released/results/bitstream_kodak3x3/kodim13_qp16.fufref2 \
  --calibration-stream proof/early_exit_vs_released/results/bitstream_kodak3x3/kodim24_qp48.fufref2 \
  --per-channel --reduce-range
python proof/cpu_early_exit/bench.py benchmark \
  --upstream proof/early_exit_vs_released/.local/DCVC \
  --extension proof/early_exit_vs_released/.local/entropy \
  --model /tmp/cpu/released_d12.onnx \
  --quant-model /tmp/cpu/released_d12_int8.onnx \
  --threads 1 --warmup 5 --repeats 30 \
  --enforce-idle --max-load 12 --out /tmp/cpu/result_t1.json
```

The default quantizer is static QDQ S8S8, Conv only. `--reduce-range` and
`--per-channel` are deliberate initial settings for this AMD EPYC 7443
(AVX2, no AVX-512 VNNI), not a quality guarantee. For selective experiments,
`--quant-selection trunk_pointwise` quantizes only the main 1×1 trunk
convolutions, and `first6_pointwise` limits those to the first six positions.
Use a different `--quant-model` path for each candidate. Calibration streams
must be supplied explicitly and come from images disjoint from evaluation.
Two Kodak streams serve only as a smoke test; an INT8 result needs
representative held-out calibration and validation images across QPs before
it can support a paper claim.

On an otherwise idle server, use the same pinned streams and compare paired
1/2/4/8/16-thread runs, preserving the JSON with all raw timings. Pin the
process to one NUMA node first, then explicitly compare node-local versus
cross-node placement. The present server is dual-socket with 96 logical CPUs
and heavy concurrent training load, so exploratory timings are diagnostics,
not speed claims. CPU benchmarking must not preempt the training jobs.

Optimization order is determined by profiling. The decoder has twelve
`DepthConvBlock` trunk blocks, but each block is dominated by dense 1×1
channel mixing; the 3×3 depthwise convolution is a tiny share of its MACs.
Consequently a hand-written depthwise AVX kernel is a poor first target.
The next implementation should use the oneDNN EP or oneDNN primitives for
pointwise convolutions and inspect post-op fusion (activation and residual
sum). A custom C++/AVX2 kernel is justified only if a measured memory-bound
activation/add or depthwise tail remains after that. AVX-512 intrinsics must
not be compiled for this host because the CPU does not support AVX-512.

Official references: [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
[ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[oneDNN EP](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html),
[ORT build options](https://onnxruntime.ai/docs/build/eps.html),
[oneDNN post-ops](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html).
