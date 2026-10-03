# Early-exit GPU kernel audit

`raw/` contains unmodified copies of six GPU measurement records and one
CPU router-map audit from the FLEX-PLUS `results/` directory. Run `python3
scripts/triton_gpu_audit_20261003.py` from the paper root to recompute
`audit.json` and `figs/triton_20261003/latency_audit.pdf` without a GPU.
The audit verifies every displayed median and paired ratio against the raw
samples. Source SHA-256 values are included in `audit.json`.

These measurements use an NVIDIA RTX A6000 shared with training. The timed
region is decoder synthesis only for one CTC first frame at QP32. The final
kernel output was checked at five real CTC/QP/map cases. The records do not
establish isolated-GPU cohort latency or complete-codec speedup.
