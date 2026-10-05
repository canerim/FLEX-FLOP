# Frozen third-cohort active-beta transfer

The local `data/clic` collection contains 41 PNGs. Before the
active-replicate β calibration or its DIV2K validation finished, we froze
all **39** images that can supply the same centred 768×512 crop used in
the DIV2K experiment. Two images are excluded solely because they are
smaller than that crop; their names and source hashes are preserved in
`proof/cpu_early_exit/results/clic39_active_beta/manifest.json`. Source
quality, router decisions and model reconstructions did not influence
image selection. All five QPs (0/16/32/48/63) are planned: 195 cases.

For each source crop/QP, encode one FUFREF2 research bitstream with the
released D12 analysis/entropy path, decode its latent, then run the same
e15 checkpoint under: (a) full-frame e15, (b) deployed isolated tiles
with their trained repair at the original DIV2K-locked β, and (c)
active-replicate/no-repair with the newly locked calibration-only β.
Also replay active-replicate/no-repair at the original β. This fourth arm
separates the context intervention from β retuning on each CLIC image.
The streams, source crop, checkpoint, router scores and rate remain fixed
between decoder arms. Record actual Δ444, YUV 6:1:1 reconstruction PSNR,
analytical synthesis-conv MAC, exit maps and stream/source hashes. If
the quality curves have sufficient common rate support, calculate
per-image BD-rate against the released D12 reconstruction; otherwise
report the exact QP points without extrapolation. Compute the same
per-image BD-rate against the deployed old-β arm as a paired operational
comparison. PCHIP fits log-rate as a function of YUV 6:1:1 PSNR only on
strictly monotone five-QP curves with measured common quality support;
record every exclusion reason. Summarize by QP and bootstrap over images,
preserving the paired arms. No CLIC outcome may change either β policy.

The frozen manifest and runner live at
`proof/cpu_early_exit/results/clic39_active_beta/manifest.json` and
`proof/cpu_early_exit/run_clic39_active_beta.sh`. The runner waits for the
DIV2K tail-risk chain to finish, checks its completed raw output, then
runs a one-case smoke, the remaining 194 cases and the analysis on one
low-priority CPU thread with CUDA hidden. This sequencing protects the
ongoing training and the higher-priority validation chain.

This cohort is separate from the DIV2K β-fitting and β-validation images.
It is **not** a previously untouched project dataset: some of these CLIC
files appeared in earlier work on other decoders. It is therefore a
third-content transfer for this DCVC-UF early-exit protocol, not a claim
of blind external evaluation. The CPU study is untimed and does not prove
native DCVC-UF bitstream compatibility or full-codec speedup. The
two-image geometry exclusion must remain visible in any paper report.
