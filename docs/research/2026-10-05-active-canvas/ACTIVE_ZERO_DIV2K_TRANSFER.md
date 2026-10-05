# Active-zero transfer to 24 DIV2K validation crops

The stage-synchronous active-zero decoder has completed all 24 DIV2K
validation crops at QP 0, 16, 32, 48 and 63: 120/120 frozen FUFREF2 streams.
The streams, DIV2K-calibrated beta and exit maps, e15 checkpoint and
full-frame e15 reference are unchanged. The stream and source hashes were
checked during each replay. This is an untimed CPU FP32 reconstruction
experiment; the records contain YCbCr 4:4:4 loss against full-frame e15,
not five-point YUV rate–distortion curves or released-D12 quality.

| Decoder path | Mean gain over deployed | 95% image-cluster interval | Cases >0.1 dB below full-frame e15 |
|---|---:|---:|---:|
| Deployed isolated tiles with trained repair | anchor | — | 35/120 |
| Active-zero with same repair | +0.00646 dB | [−0.00012,+0.01261] dB | 31/120 |
| Active-zero without repair | **+0.01695 dB** | **[+0.01158,+0.02268] dB** | **24/120** |
| Exact-context diagnostic without repair | see raw exact-context audit | — | 24/120 |

Against deployed, the no-repair arm rescues 11 over-threshold cases and
creates none. The matched-repair arm rescues five but creates one new
over-threshold case. Comparing the two active-zero reconstructions *directly*,
removing the trained repair improves mean 4:4:4 PSNR by +0.01049 dB
(24-image clustered interval [+0.00499,+0.01698] dB); 90 cases improve
and 30 worsen. This repair contrast is **exploratory**: it was highlighted
after inspecting the complete active-zero transfer, whereas the frozen
primary transfer comparison is active-replicate versus active-zero with
the repair held fixed. The same Kodak result had favored no repair, but
Kodak also informed development and is not an untouched confirmation set.

The mechanism is plausible: the existing seam-repair weights were trained
for isolated-tile boundary errors and can overcorrect after real active
neighbour context is supplied. This explanation requires a targeted
repair retraining or spatial residual analysis; the scalar data alone do
not prove it. Active-replicate transfer is still running, so this note does
not choose a deployment fallback. The PyTorch halo path is not timed.

There is a separate arithmetic consequence on the frozen Kodak maps:
the audited active-zero halo adds only 0.01087 percentage points of
full-synthesis convolution MAC, while removing grid repair saves 0.95072
points. Thus active-zero **without** repair has 26.82359% analytical
synthesis-convolution saving versus 25.88374% for deployed isolated tiles.
This is a +0.93985-point cost-model improvement, not a measured latency gain
or a DIV2K-specific MAC measurement. The [120-case cost audit](../../../proof/cpu_early_exit/results/kodak_active_canvas_cost_20261005.json)
contains the fixed-map calculation.

Source: [`active_canvas_validation24_zero.json`](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_zero.json)
and its [hash-linked clustered summary](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_zero_summary.json).
