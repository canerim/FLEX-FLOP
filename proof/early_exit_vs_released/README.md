# e15 early exit versus released D12

## Reproduce from a fresh clone

The proof branch contains the released D12 and e15 checkpoints as Git LFS
objects and the 585-KB decoder-side router checkpoint in ordinary Git. The
checkpoint SHA-256 values are checked before building or benchmarking. On a
Linux NVIDIA GPU with a CUDA 12.6-compatible driver, Python 3.12, `g++`, Git
and Git LFS:

```bash
git clone --branch proof/early-exit-vs-released-20261003 \
  https://github.com/canerim/FLEX-FLOP.git
cd FLEX-FLOP
git lfs pull
python3.12 -m venv proof/early_exit_vs_released/.local/venv
proof/early_exit_vs_released/.local/venv/bin/python -m pip install --upgrade pip
proof/early_exit_vs_released/.local/venv/bin/python -m pip install \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
proof/early_exit_vs_released/.local/venv/bin/python -m pip install \
  numpy==2.2.6 scipy==1.18.1 pillow==11.3.0 pybind11==3.1.0
proof/early_exit_vs_released/.local/venv/bin/python \
  proof/early_exit_vs_released/bootstrap.py
proof/early_exit_vs_released/.local/venv/bin/python \
  proof/early_exit_vs_released/bitstream_benchmark.py benchmark \
  --stream proof/early_exit_vs_released/results/kodim01_qp32.fufref2 \
  --source data/kodak/kodim01.png --gpu 0 --blocks 20 \
  --save-recon-dir proof/early_exit_vs_released/results/reconstructions \
  --out proof/early_exit_vs_released/results/kodim01_reproduced.json
```

`bootstrap.py` checks all three artifact hashes, sparse-clones Microsoft DCVC
at commit `cbdae87a5445114cdc7f48816da63ea80bdeac40`, and builds the pinned
CPU rANS extension locally. `bitstream_benchmark.py prepare --image PNG --qp 32
--out FRAME.fufref2` can encode another RGB PNG whose dimensions are multiples
of 256. Benchmark output contains paired raw timings, checkpoint and stream
hashes, router exit counts, output equivalence, source quality, GPU occupancy
checks, and code/environment identities. The optional PNG files show the actual
released and early-exit outputs and are written outside timed blocks. It refuses
an occupied GPU. The sample
stream was emitted from the tracked Kodak image, is 16,627 bytes, and has SHA-256
`ae06007c9ebef84894b8aacaaf7d551a9a062076aac97590586a42652d609ceb`.
The [public reproducibility check](PUBLIC_REPRO_CHECK_20261003.json) records a
separate clean checkout and unauthenticated HTTPS download of both LFS weights.

For the stricter image-to-image question, add `--include-encoder --blocks 5` to
the benchmark command. Each paired arm then independently encodes the source
PNG into the same checked bytes before decoding. This is substantially slower
to measure because the research encoder runs on CPU; report it separately from
the bytes-to-image decoder result.

For a predeclared cross-content check, nine tracked streams cover Kodak
`kodim01`, `kodim13`, `kodim24` at QP 16/32/48. The
[`manifest.json`](results/bitstream_kodak3x3/manifest.json) fixes source, stream
and decoded-latent hashes before GPU results are observed. After setup:

```bash
proof/early_exit_vs_released/.local/venv/bin/python \
  proof/early_exit_vs_released/run_bitstream_cohort.py --gpu 0 --blocks 20
proof/early_exit_vs_released/.local/venv/bin/python \
  proof/early_exit_vs_released/run_bitstream_cohort.py --gpu 0 --blocks 5 \
  --include-encoder --out proof/early_exit_vs_released/results/roundtrip_cohort
```

`prepare_bitstream_cohort.py` regenerates all nine streams on CPU and refuses
an existing file with different bytes. The runner records every case and its
raw paired samples; its aggregate is the median of the nine *case medians*,
with the observed case range. Three images are a fixed workload set, not a
population confidence interval or a Kodak-wide claim. The idle-GPU watcher on
the research server queues matched-kernel synthesis, this bytes-to-image
cohort and then the full image roundtrip cohort, in that order.

**Meaning of the numbers:** the fresh-clone test starts with in-memory
`FUFREF2` bytes; each arm independently does CPU rANS and hyperprior decode,
copies the latent to the GPU, then synthesizes the image. The e15 arm also runs
its stem+QP router on the decoder side and reuses that stem during synthesis.
This is a genuine bytes-to-output decoder test for the **research wire format**.
The encoder, disk I/O and model load are outside timed blocks. `prepare` reports
CPU encoder time separately. The format is not Microsoft's native CUDA stream;
results may not be called native DCVC-UF throughput. Kodak quality is YUV 6:1:1
PSNR in 4:4:4, which is distinct from the CTC YUV420 metric. The existing
2.758× number below remains synthesis-only until the new paired result exists.

The [research note](RESEARCH_20261003.md) gives the GPU-free experiment priorities
and the reasons for each runtime control.

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
