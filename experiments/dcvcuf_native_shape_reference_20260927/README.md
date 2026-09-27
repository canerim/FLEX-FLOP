# Native-shaped CPU reference: a separate padding-policy control

Microsoft's pinned CUDA path pads the source image to a multiple of16 and
replicate-pads the analysis latent to a multiple of4 before hyperanalysis.
The earlier FUFREF1 CPU reference instead pads the whole image to64. Both
policies coincide at512,256 and320, but differ for288-pixel context windows.
This distinction was identified before either patch experiment produced results.

`prepare.py` derives a separate source snapshot under the experiment root's
`research/native_shape_reference/source`. It changes only the image padding,
hyperanalysis latent padding, decoder shape derivation and format magic.
The existing running reference files remain unchanged. Base and derived source
hashes and native source evidence are in `preparation.json`.

**FUFREF2 is still a CPU FP32 research format.** It uses the same causal rANS
reference, strict symbol checks, no threshold skipping, and88-byte integrity
header. It does not claim compatibility with the released CUDA bitstream,
FP16 reconstruction, native scale-index calculations or CUDA timing.

`verify.py` passed45 cases: D2/D6 epoch20 and releasedD12, QP0/32/63,
and geometries64²,65×97,288²,320²,512². Every stream was decoded in a new
source-disabled decoder process with matching symbols, indexes, latents and
reconstruction. The27 cases aligned to64 also matched FUFREF1 payloads and
reconstructions exactly. These are engineering checks, not benchmark quality.

`evaluate_padding.py` waits for the complete primary240-case patch study.
It uses the same16 predeclared images, five QPs and three depths, and re-encodes
only the halo32 windows under the16/4 policy. Full512 reconstruction is checked
again after version-magic rewrapping; aligned core256 and halo64/context320
results remain reusable, with original records retained. No source images or
QP points are selected based on outcomes. The primary study and this paired
control must retain distinct provenance and labels.

```bash
RUN_ROOT=/data10/shareddata/can_karsal/dcvcuf_depth_20260927
# Preparation and verification already ran; they refuse accidental overwrite.
# Evaluator is currently queued and must not be started a second time.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  nice -n19 "$RUN_ROOT/venv/bin/python" \
  experiments/dcvcuf_native_shape_reference_20260927/evaluate_padding.py
```

Result path: `research/patch_native_shape_epoch020` under `RUN_ROOT`.
No active training source, optimizer, recipe or GPU allocation is changed.
