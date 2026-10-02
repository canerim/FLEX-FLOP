# QP-conditioned value of added synthesis blocks

CPU-only re-analysis of the pinned `data/refresh20260927/exit_depth_profile.json`
archive. The input contains 53 sequences × 5 QPs × 3 adjacent depth pairs,
from the **actual uniform-exit reconstructions** of the shared-exit e15
model. The network input is padded YCbCr 4:4:4, despite historical `RGB`
field names in the archive; `data/refresh20260927/metric_provenance.json`
documents this naming issue. No training, inference or GPU allocation occurs.

`python3 scripts/qp_depth_value_20261002.py` validates the full grid and
calculates mean within-frame median tile gains, sequence-cluster confidence
intervals and paired QP63−QP0 changes. It reproduces the one-column PDF/PNG
figure. Its comparisons change all tiles' depths at once; they are not causal
single-tile interventions or an independently trained codec-bank analysis.
