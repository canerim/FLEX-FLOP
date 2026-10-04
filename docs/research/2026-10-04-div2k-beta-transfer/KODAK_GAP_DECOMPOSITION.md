# Where the Kodak YUV611 gap to released D12 comes from

The exact-context Kodak replay reduced the mean released-to-deployed gap
from 0.1382 to 0.1208 dB, leaving most of the gap unexplained. We measure
full-frame epoch-15 e15 YUV611 on the **same 120 actual FUFREF2 streams**
and split the per-case gap into an exact additive identity:

`released - deployed = (released - e15 full) + (e15 full - routed exact context) + (routed exact context - deployed)`.

The terms distinguish checkpoint/training difference, depth routing under
exact context, and the combined effect of deployed tile context plus learned
repair. Each uses the same source, stream, QP, and cropped frame; direct
checks verify archived full-frame MSE and the equality to numerical
tolerance. The first term can be negative because e15 and released have
different trained synthesis weights. The third term is not solely a padding
effect: the diagnostic exact variant also disables the old repair.

## Results

All **120/120** CPU replays completed and passed the archived full-frame
MSE and additive-identity checks. The mean released-to-deployed YUV611 gap
is **0.13816 dB**. Its signed components are:

| Cause (same stream/source) | Mean gap contribution | 24-image bootstrap 95% interval |
|---|---:|---:|
| Released D12 to full-frame e15 checkpoint | 0.01937 dB | [0.01663, 0.02265] |
| Full-frame e15 to fixed-route exact context | **0.10140 dB** | [0.08702, 0.11525] |
| Exact context to deployed zero-halo plus repair | 0.01740 dB | [0.01405, 0.02090] |

The route/capacity term contributes **73.4% of the mean gap**
(0.10140/0.13816); the checkpoint and context/repair terms contribute
14.0% and 12.6%, respectively. These are ratios of signed cohort means,
not a claim that every individual gap has the same decomposition.

| QP | Checkpoint | Fixed route | Context/repair | Total released-to-deployed gap |
|---:|---:|---:|---:|---:|
| 0 | -0.0003 | 0.1173 | 0.0145 | 0.1315 |
| 16 | 0.0054 | 0.0722 | 0.0177 | 0.0952 |
| 32 | 0.0139 | 0.0935 | 0.0172 | 0.1246 |
| 48 | 0.0294 | 0.1258 | 0.0154 | 0.1706 |
| 63 | 0.0485 | 0.0982 | 0.0222 | 0.1689 |

At QP0 the checkpoint contribution is essentially zero on average, while
it rises to 0.0485 dB at QP63. The fixed-route term remains the largest
mean cause at every QP. This directly prioritizes better depth allocation
and shallow-exit capacity over context engineering alone. It does **not**
establish an achievable speed/quality frontier: the exact-context variant
has no sparse implementation or measured end-to-end latency.

![Three-stage Kodak gap at actual bitrates](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/kodak24/exact_context_kodak.png)

## Reproduce

Run `proof/cpu_early_exit/decompose_kodak_released_gap.py` with the released
analysis/entropy and e15 checkpoint/extension already pinned in the Kodak
replay. It reads the frozen exact-context JSON and checks source, scan,
transfer, and bitstream hashes before every case. Use `CUDA_VISIBLE_DEVICES=''`
and a single low-priority CPU thread. No GPU training state is touched.
