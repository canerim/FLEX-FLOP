# Calibration-only tail-risk beta for active-context early exit

The mean Δ444 ≤0.1 dB policy can hide individual losses above 0.1 dB.
This separate exploratory arm asks how much analytical synthesis-conv
saving is sacrificed to constrain the *calibration* tail. It leaves the
primary beta choice, FUFREF2 streams, e15 weights and validation cohort
unchanged. The rule was recorded before the complete active-context
calibration or its validation transfer was available.

At each QP, `select_active_beta_tail_policy.py` considers only archived
beta candidates on the 24-image DIV2K calibration partition. A candidate
is eligible if its **mean Δ444 ≤0.1 dB** and **at most 2/24 calibration
images lose >0.1 dB** against full-frame e15. Among eligible candidates,
choose the greatest mean analytical synthesis-conv MAC saving; ties
minimize mean loss, number of violations and beta. Thus the two policies
share a mean target, while this one additionally controls the empirical
tail. No validation or Kodak datum enters selection. If a QP has no
eligible candidate, the selector fails visibly rather than relaxing the
rule.

`audit_active_beta_tail_validation.py` replays the locked policy on the
separate DIV2K validation24 partition, checking unchanged source hashes,
bitstreams and selected exit maps against the primary mean-beta replay.
`analyze_active_beta_tail_validation.py` reports paired mean quality and
MAC differences, 0.1-dB failure counts, rescued/new failures and
24-image clustered uncertainty. The policy is not retuned if the
calibration tail constraint fails on validation.

The work is CPU FP32 and untimed. The MAC saving is a synthesis-conv
arithmetic model, not full-codec latency; Δ444 against e15 is not YUV
BD-rate against released D12. DIV2K validation has informed earlier
project analyses, so this is an exploratory transfer rather than an
untouched final benchmark.
