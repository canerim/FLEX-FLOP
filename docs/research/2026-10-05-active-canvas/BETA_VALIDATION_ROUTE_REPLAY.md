# Validation route reconstruction for newly locked beta

The original DIV2K validation-case archive contains one candidate per
image/QP: the previously locked beta. The new active-replicate
calibration selected beta 45 at QP0 and beta 5 at QP63, so looking up
those prices in the archived validation cases correctly failed before
any validation result was written. This was a protocol implementation
error, not a failed decoder reconstruction.

The validation replay now loads the frozen stem/QP router checkpoint and
calibration, decodes the unchanged FUFREF2 latent, recomputes log scores
from the e15 full-frame stem, and applies the locked price with the same
cost vector as calibration. For every case it first recomputes the old
price and requires its exit map and analytical cost to match the archived
single candidate. Only then does it decode the newly selected map.
The first QP0 case passed this gate and reconstructed at beta 45.

The same route helper is used by the later budget and tail-policy
validation jobs. No validation output enters the calibration-only price
selection, and none of the original streams, source crops or model
weights are changed. The CPU jobs remain sequential and low-priority;
the lower-priority seam-locality replay was stopped after a small
resumable prefix and requeued behind the primary beta analysis.
