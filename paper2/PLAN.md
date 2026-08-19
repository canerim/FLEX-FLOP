# paper2 — the reviewer's list

Seven things a CVPR reviewer would attack, and what closes each. Tracked apart
from the main paper because most of them need new measurements rather than new
prose.

Status is updated as work lands. Nothing here is claimed until it is measured.

| # | objection | what closes it | status |
|---|---|---|---|
| 1 | "Why not train under the coupled routed setting?" | replicate-trained/coupled-inference vs coupling-trained/coupled-inference. A negative result is still worth having: it turns "training mismatch" into "structural dependency". | not started |
| 2 | "Why are we evaluating an unfinished model?" | train to convergence, refresh the main tables from the final checkpoint, and put a saving / PSNR / val-loss vs step curve in the supplement. The sentence "training is not converged" leaves the paper. | MSR45 at epoch 38/45 |
| 3 | Tile definition is not explicit | four lines in Method: N = ceil(W/P) ceil(H/P), whether an edge tile is cropped or padded, how the feature-space tile size follows, and where the padding is applied. Then a resolution-adaptive tile-size experiment: 128 px for HEVC-C/D, 256 px above. | not started |
| 4 | "Then why is the learned router there?" | router input ablation: stem only / latent only / scales only / bits only / all. Close the mechanism instead of asserting it. If the free rule wins, make that a contribution rather than a footnote. | not started |
| 5 | "Is this general or a DCVC-UF observation?" | a second decoder, minimally: seam scaling and early-exit feasibility, two or three exits, one quality point. No oracle, no router. Failing that, narrow the claim to DCVC-UF-like residual decoders. | not started |
| 6 | Scope: a video-codec paper that only shows intra | say it in the Introduction, early and plainly: the intra synthesis path is the controlled setting and temporal propagation is deliberately out of scope. Then one small supplement experiment: 0.1 dB of I-frame degradation, measured through the next N frames. | not started |
| 7 | Figures 2 and 7 are below the standard of the rest | redraw to the standard of Figure 9c. | not started |
