# D8/D10/D12 training health — 4 October 2026

This is a CPU-only audit of the live official-recipe runs. The training jobs, checkpoints, optimiser state and GPU execution were not modified. The accompanying vector figure interpolates the **same four fixed DIV2K centre crops** to **0.2 estimated bpp** and reports mean YUV611 PSNR. It is a useful health monitor, not Kodak RD or actual bitstream quality.

| Model | Live epoch at audit | Latest monitor | Best monitor so far | Nonfinite skips |
|:--|--:|--:|--:|--:|
| D8 | 76/105 | 38.074 dB | 38.074 dB (epoch 76) | 0 |
| D10 | 66/105 | 37.973 dB | 38.004 dB (epoch 65) | 0 |
| Scratch D12 | 43/105 | 37.606 dB | 37.606 dB (epoch 43) | 0 |

At the matched 40th epoch, D8/D10/D12 give 37.467/37.508/37.576 dB. At epoch 42 they give 37.441/37.478/37.369 dB. **D12 rebounds to 37.606 dB at epoch 43, a new best on this monitor**; its QP63 YUV611 PSNR likewise recovers from 40.631 to 41.219 dB. The dip was real on the fixed samples but was not a persistent collapse. D12 is still in the official **2×10⁻⁴ LR phase through epoch 44**. D8 and D10 also fluctuate in this phase and then rise sharply after the official LR reduction at epoch 45; by epoch 50 they reach 37.924 and 37.973 dB. D8 set a further monitor best of 38.074 dB at epoch 76; D10's 37.973 dB at epoch 66 is 0.031 dB below its epoch-65 best, a single-epoch fluctuation rather than a demonstrated deterioration. The 512-pixel crop phase begins at epoch 90 and has not started for any of these runs.

All three status files were fresh, their processes were alive and producing optimizer updates, and no nonfinite batches were skipped. A single instantaneous `nvidia-smi` sample can show low utilisation during data/checkpoint work; the status step counters are the more reliable liveness check. A concern would be **persistent deterioration after D12 crosses the 45-epoch LR transition**, ideally checked on a broader held-out image set and at actual bitstream rates. Stopping or changing LR now would violate the official recipe and invalidate the direct depth comparison.

The plot and machine-readable summary are `active_depths_health.pdf` and `active_depths_health.json` in this directory. Recreate them with `experiments/dcvcuf_depth_official_20260927/analyze_active_depths.py --runs /data10/shareddata/can_karsal/dcvcuf_depth_20260927/runs --output docs/research/2026-10-04-active-depth-health`.
