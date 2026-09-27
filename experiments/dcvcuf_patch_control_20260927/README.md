# Fixed-depth patching control

This experiment isolates the cost of independent patch coding before a model
selector is introduced. It uses frozen epoch20 D2/D6 and released D12, five
QPs, and 16 validation images selected by evenly spaced IDs from 0801–0900.
The image list is fixed in the script before results are observed.

Each 512×512 centre crop is compared with four 256×256 cores encoded either
independently or with 32/64 pixels of context. Context windows are clipped to
the 512 crop, giving 288×288 or320×320 windows for this 2×2 grid. The codec pads
both halo variants to320×320 internally. Only its core reconstruction is retained;
there is no overlap blending. All payload bits, including overlap/context
coding, count against the original 512×512 area.

The full-crop streams are reused from the completed 2,000-case reference
validation. Every patch has its own real research-format entropy stream,
decoded in an isolated process. The comparison is fixed-depth throughout;
it contains no router or expert-selection gain. The 88-byte per-stream
research header is reported separately from payload.

Outputs include same-QP rate and quality changes and MSE in 4/16-pixel
bands around the two internal seams. Same-QP changes are descriptive;
matched-rate comparisons require the completed five-QP curves. CPU times
are not deployment latency. Released D12 has different training provenance
from the shallow models.

The 256,288 and320 input geometries passed an independent-process decode
preflight on D2/QP32. The seam-mask areas and constant-error metric were
checked analytically. Full evaluation is launched only after the large
reference run completes, keeping CPU jobs sequential on this busy server.

```bash
RUN_ROOT=/data10/shareddata/can_karsal/dcvcuf_depth_20260927
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 10 "$RUN_ROOT/venv/bin/python" \
  experiments/dcvcuf_patch_control_20260927/evaluate_patching.py
```

Results are stored under `$RUN_ROOT/research/patch_control_epoch020`.
The evaluator refuses to start unless the full reference run is complete.

## Before-outcome amendment,27 September17:54UTC

Halo64 was added before the patch evaluator started or produced any outputs.
The hypothesis follows from padding geometry: it supplies more actual context
than halo32 at exactly the same coded area. It also aligns inner window starts
to the64-pixel hyperlatent grid. This does not ensure equal payload, quality
or runtime. The original waiting-queue state and preflight were archived.
A fresh256/288/320 geometry and independent-decoder preflight passed
at17:55:27UTC; the sequential queue restarted at17:56:13UTC. Matched-rate analysis uses common
support across all four variants, with coverage reported.
