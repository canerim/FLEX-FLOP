# Active-zero Kodak YUV BD-rate at fixed research bitstreams

This calculation uses all 24 Kodak images and five QPs per image. Each arm
decodes the **same frozen FUFREF2 bitstream** at each image/QP; its rate is the
complete research-stream bytes times eight divided by source pixels. YUV
6:1:1 PSNR is recomputed for each reconstruction. Per-image BD-rate uses
PCHIP on log(rate) over the common PSNR support, then the 24 image values
are averaged. A separate PCHIP integral matches the `bjontegaard` package
to within 1e−10 percentage points for every curve.
All 120 research stream lengths and SHA-256 digests were checked against the
case records before using their rates.

| Decoder path | Mean BD-rate vs deployed | 95% image-bootstrap interval | Mean BD-rate vs released D12 |
|---|---:|---:|---:|
| Deployed isolated tiles, repair | anchor | — | +3.665% |
| Active-zero context, same repair | **−0.244%** | [−0.342, −0.143]% | +3.411% |
| Active-zero context, no repair | −0.367% | [−0.440, −0.295]% | +3.285% |
| Exact-context diagnostic, no repair | −0.445% | see raw analysis | +3.203% |
| Legacy stale canvas, same repair | +0.662% | see raw analysis | +4.355% |

With repair, 21/24 images have lower BD-rate than deployed; three
(`kodim10`, `kodim16`, `kodim21`) have higher BD-rate. Without repair all
24 images have lower BD-rate, but that comparison changes both context and
repair. The matched-repair arm isolates the effect of changing depthwise
context at a fixed route, checkpoint and stream. Its mean improvement is
small, and a +3.411% gap to released D12 remains. The gain does not establish
equal-quality routing superiority over a blind depth mixture, native
DCVC-UF bitstream interoperability, or full-codec runtime speedup.

The active-zero rule followed a five-case pilot; this Kodak result is
exploratory. The active-replicate and DIV2K transfer arms should decide
which fallback, if any, transfers. The current PyTorch implementation has
untimed halo packing and no native CUDA stream. Do not promote this BD-rate
number to a deployment claim until those measurements are complete.

Reproduce with
`python proof/cpu_early_exit/analyze_canvas_bdrate.py --arms stale active_zero --output proof/cpu_early_exit/results/kodak_canvas_bdrate_active_zero_20261005.json`.
The [per-image curves and source hashes](../../../proof/cpu_early_exit/results/kodak_canvas_bdrate_active_zero_20261005.json)
are retained in the result file.
