# e15 early exit versus released D12

`benchmark.py` compares the Microsoft released synthesis network to the trained e15 early-exit network, with and without the inference-only Triton patches. The source is the first `videoSRC05` CTC frame, padded identically. QP and the archived budget-0.1 router map are fixed for each run. All arms decode **one identical latent**; the script refuses to proceed unless every non-decoder checkpoint tensor in e15 equals the released tensor exactly. This isolates synthesis. The result does not include analysis encoding, hyperprior/rANS, router decision, stream I/O, or network transfer. It is therefore a decoder-synthesis speedup, not an end-to-end codec speedup.

**Interpretation of the existing 2.758× result:** released D12 runs with stock PyTorch operators, while the fastest e15 arm uses custom Triton fusion. The nine-workload median released/e15-stock paired speedup is 1.262×; that PyTorch comparison still includes tile scheduling, so it is not a pure architecture ablation. MAC reduction alone cannot be credited with the full 2.758×. This repo now has a `--matched-kernels` control that applies the same FFN, pointwise, depthwise and activation fusions to the released D12 structure. It also runs e15 with **all tiles at full depth** under both PyTorch and Triton, giving a within-model, within-scheduler measurement of the saving from early exits. The existing 2.758× measurements predate these controls; no matched-kernel or isolated early-exit speedup is claimed until a new idle-GPU cohort is recorded. The D12 installer has passed a CPU output-equivalence check, but its GPU output and speed still require measurement.

Each arm receives warmup, then twenty or more interleaved randomized paired blocks. CUDA-event and synchronized wall times are saved for every block. The wall-time median of the per-block released/e15+Triton ratios is the main speedup. The output includes all raw observations, PSNR on the valid YUV420 source, patch output error, CUDA/PyTorch versions, model hashes and map counts. The script **fails closed if another process is on the selected GPU**. A diagnostic override exists for development, but the resulting JSON explicitly marks it ineligible for a speedup claim. `summarize.py` requires all nine combinations of three fixed first frames (1080p, 720p, 480p) and QP 16/32/48 on an idle GPU before producing a cohort-level number. QPs from one frame are correlated, so the cohort reports the observed workload range rather than a population confidence interval.

```bash
cd /home/can_karsal/FLEX-PLUS
.venv/bin/python proof/early_exit_vs_released/benchmark.py --gpu 0 --qp 32 --blocks 20
.venv/bin/python proof/early_exit_vs_released/run_cohort.py --gpu 0 --blocks 20
.venv/bin/python proof/early_exit_vs_released/run_cohort.py --gpu 0 --blocks 20 --matched-kernels
.venv/bin/python proof/early_exit_vs_released/server.py
```

`run_when_idle.py` can wait for a completely unused GPU (no compute process,
under 256 MiB allocated, under 5% utilization) for five continuous minutes,
then run the matched-kernel cohort once. It polls only `nvidia-smi` while
waiting and expires after 24 hours by default. The benchmark aborts if another
compute process arrives during its paired measurements. The watcher's JSONL
and run log are written under `results/`.

`mac_latency_audit.py` reconciles the recorded padded shapes and tile maps with
convolution MACs. It includes the trained early-exit adapters and seam repair;
host planning and pointwise activations have no convolution MAC count. A CPU
forward-hook test checks the released model's analytic formula. This audit
must be used instead of multiplying the nominal source resolution by the
decoder's per-pixel MACs, because the benchmark pads to whole tiles first.

Open the local address printed by the server and enter the printed run token. The browser shows a live comparison after each paired block. The server accepts only fixed workloads and starts at most one benchmark at a time. It binds to loopback by default; a public deployment should place it behind authentication and use `PROOF_RUN_TOKEN` for the run endpoint. Do not expose an unauthenticated GPU benchmark endpoint on a shared server.

The prior e15-vs-e15 kernel audit lives in [`cvpr2027/data/triton_20261003`](../../cvpr2027/data/triton_20261003/README.md). Its shared-GPU timings cannot establish the released comparison.
