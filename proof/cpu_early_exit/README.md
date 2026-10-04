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

The [Kodak3x3 routed audit](results/kodak3x3_routed) runs three predeclared
images at QP 16/32/48 from real research bitstreams. Its
[`plot_qp_cohort.py`](plot_qp_cohort.py) verifies source and stream hashes,
computes the project's frame-level synthesis MAC model, and regenerates the
three-panel figure in `cvpr2027/figs/cpu_qp/fig_cpu_qp_route_quality_mac.pdf`:

```bash
python proof/cpu_early_exit/plot_qp_cohort.py \
  --cohort proof/cpu_early_exit/results/kodak3x3_routed
```

Across the three images, QP16 uses exits 3/4 at 66.7/33.3% and saves 20.5%
analytical synthesis MAC at 0.074 dB mean YUV 4:4:4 loss. QP32 and QP48
use the deepest exit for 83.3% and 88.9% of tiles, saving 2.6% and 1.2%.
The analytic MAC model includes adapters and seam repair but excludes the
router. None of these are CPU latency measurements. The full raw records and
manifest hashes are checked by the plot generator.

The 3x3 check is superseded for allocation claims by the complete
[Kodak24 x QP5 audit](results/kodak24_qp5): 120 independently decodable
FUFREF2 streams, source/stream hashes, decoder-side router decisions,
assembled reconstructions and analytical MAC records. The matched
[fixed-beta quality replay](results/kodak24_fixed_beta_quality) recomputes
the QP16-beta counterfactual from those same bytes and also measures both
policies against e15 full-frame output using the paper's equal-channel
YCbCr 4:4:4 MSE-ratio loss (`Delta444`). Rebuild the summary and vector plot:

```bash
python proof/cpu_early_exit/summarize_kodak_full_scan.py \
  --cohort proof/cpu_early_exit/results/kodak24_qp5 \
  --fixed-cohort proof/cpu_early_exit/results/kodak24_fixed_beta_quality \
  --out-prefix cvpr2027/figs/kodak24_qp/fig_kodak24_qp_policy
```

With equal weight for each image and QP, the CTC-calibrated policy saves
17.16% analytical synthesis MAC at 0.0565 dB mean Delta444. Holding beta
at its QP16 value saves 26.82% but costs 0.1223 dB. Respectively 14/120
and 55/120 frame--QP cases exceed the *descriptive* 0.1 dB threshold;
the CTC calibration only targeted mean loss, not per-image feasibility.
The separate released-D12-relative weighted YUV PSNR differences are
not the same metric and cannot be used to evaluate that target. Holding
beta fixed does not remove QP from the router head or latent statistics.
These numbers are analytical synthesis MAC and reconstruction quality,
not CPU or full-codec speedups. The [audit note](../../docs/research/2026-10-04-kodak24-router-transfer/REPORT.md)
separates the CTC nominal sweep from this external control-transfer test.

`run_when_idle.py` waits for five consecutive one-minute load checks below
its threshold, then records paired released/routed/all-deep PyTorch CPU
latencies at 1/2/4/8 threads over all nine streams. It pins each child to
node-0 CPUs and uses low scheduling priority; the benchmark aborts if the
host becomes loaded. This is the shared-backend CPU control, not the final
oneDNN or custom-fusion arm.

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
