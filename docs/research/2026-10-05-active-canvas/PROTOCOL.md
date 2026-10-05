# Active-neighbour context for the shared early-exit decoder

## Question and fixed comparison

Can the existing six-exit e15 decoder recover tile-boundary quality **without retraining, changing its route, or changing its FUFREF2 entropy payload** by sharing 3×3 depthwise context only between tiles that are still at the same suffix stage?

The Kodak24 × QP0/16/32/48/63 cohort and DIV2K-calibrated β policy were fixed before this experiment. The comparison uses the exact same six-exit map, full-frame e15 checkpoint, source, latent, and output metric in every arm:

1. **Deployed:** isolated tiles with replicate padding and trained grid seam repair.
2. **Stale canvas control:** existing `CanvasCoupler` reads a neighbour's last feature even after that neighbour exits. This is an instructive but potentially invalid context source.
3. **Active canvas, zero fallback:** read a neighbour only while it is still active in the current suffix group; use zero otherwise. Outer-frame padding stays zero.
4. **Active canvas, replicate fallback:** same active-only rule; when an internal neighbour has exited, use the current tile's border value, matching the deployed tile padding. Outer-frame padding stays zero.

For each coupling arm, record both no seam-repair and unchanged trained grid seam-repair output. The all-deep/no-repair reconstruction must match full-frame e15 within `atol=rtol=1e-4`. This catches halo-indexing errors without consulting source quality. Every image/QP is retained. The chosen primary diagnostic is paired Δ444 against full-frame e15, with 24-image clustered intervals; YUV 6:1:1 PSNR and counts above 0.1 dB are secondary. Computation savings are reported only as **analytical convolution MACs**; tile packing, memory traffic, entropy decoding, and real latency require separate measurement.

## Design provenance and interpretation

The active-neighbour rule was devised after a five-case pilot exposed a QP63 failure of stale coupling on `kodim01` (deployed Δ444 0.116 dB, stale no-repair 0.380 dB). On that same case active-zero gives 0.098 dB and active-replicate gives 0.0975 dB. Therefore the ensuing Kodak scan is **exploratory**, not an untouched hypothesis test; this pilot cannot select a winning variant for publication. The fallback variants are fixed here before their full-cohort outcomes. If one survives Kodak, replay it unchanged on the 24-image disjoint DIV2K validation cohort before considering a method claim, while acknowledging that DIV2K has also informed earlier project work.

The mechanism has a narrow causal interpretation: same-stage neighbours provide valid context at a 3×3 depthwise convolution; exited neighbours no longer do. This is distinct from changing the router, the depth histogram, the checkpoint, or the bitstream. An improvement in RGB/YCbCr quality at frozen maps would be evidence for an inference-path improvement, **not** for net rate-distortion or speed advantage until the implementation is timed and its costs are audited.

Kodak replay scripts: [`audit_canvas_coupling_kodak.py`](../../../proof/cpu_early_exit/audit_canvas_coupling_kodak.py), [`audit_active_canvas_kodak.py`](../../../proof/cpu_early_exit/audit_active_canvas_kodak.py), and [`audit_active_canvas_replicate_kodak.py`](../../../proof/cpu_early_exit/audit_active_canvas_replicate_kodak.py). The active-only implementations are [`active_canvas_coupling.py`](../../../proof/cpu_early_exit/active_canvas_coupling.py) and [`active_canvas_replicate.py`](../../../proof/cpu_early_exit/active_canvas_replicate.py). Each replay records hashes of the relevant decoder code, checkpoints, stream, and archived fixed-map case.
