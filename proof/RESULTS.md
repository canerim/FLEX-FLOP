# Audited results — 3 October 2026

## Shared-latent e15 early exit versus Microsoft's released D12

**Result:** on nine fixed first-frame CTC workloads (480p/720p/1080p × QP 16/32/48), the median of paired released-D12/e15+Triton decoder-synthesis speedups is **2.758×** on an otherwise idle NVIDIA RTX A6000. The nine workload medians span **2.486×–3.003×**. Each uses 20 randomized paired blocks after warmup. All nine JSON files record empty competing-GPU-process lists before, during and after timing. The checkpoint hashes are identical across the nine cases, and all **255 non-decoder released/e15 tensors match exactly**. The e15 stock versus Triton output differs by at most a few 10⁻⁷ per case. [`cohort_summary.json`](early_exit_vs_released/results/cohort_20261003/cohort_summary.json), [`raw cases`](early_exit_vs_released/results/cohort_20261003/) and [`figure`](early_exit_vs_released/results/cohort_20261003/speedup_audit.pdf) are the audit trail.

**Implementation-control caveat:** the released D12 arm uses stock PyTorch operators, whereas the fastest e15 arm uses inference-only Triton fusions. The median released/e15-stock speedup is **1.262×** across the nine workloads; the full **2.758×** must not be described as an early-exit-only or MAC-only gain. A matched-kernel released-D12 arm has been implemented but has no idle-GPU measurement yet.
The [`latency decomposition`](early_exit_vs_released/results/cohort_20261003/latency_decomposition.pdf) plots all three measured arms and makes this distinction visible.

| First CTC frame | QP16 | QP32 | QP48 | Released / e15+Triton wall ms at QP32 |
|---|---:|---:|---:|---:|
| BQMall, 832×480 | 2.852× | 2.661× | 2.486× | 30.38 / 11.41 |
| FourPeople, 1280×720 | 3.003× | 2.656× | 2.529× | 56.15 / 21.03 |
| videoSRC05, 1920×1080 | 2.917× | 2.804× | 2.758× | 146.50 / 52.16 |

At the same QP, released D12 has a mean **0.104 dB** higher YUV 6:1:1 PSNR across these nine workloads (range 0.098–0.111 dB). The latency comparison uses one exact shared latent and includes host tile planning in e15 wall time. It excludes analysis encoding, hyperprior/rANS, router inference, bitstream I/O and mode-map signaling. Thus **2.758× is a decoder-synthesis speedup, not an end-to-end codec speedup**. It also does not compare quality-matched QPs. The first-frame/three-sequence sample supports a reproducible workload claim, not a population confidence interval for all CTC frames. The browser UI at `http://127.0.0.1:8765/` shows fresh paired times live; run token is in the server terminal/log.

The nine-case median released/e15 **stock** ratio is 1.262×. The median e15 stock/e15+Triton ratio is 2.220×. These are different comparisons from the earlier shared-GPU e15 masked-stock audit and should not be multiplied as if from one experiment.

## Independent D2/D4/D6 actual-byte Kodak BD-rate

All four codecs (three final epoch-105 depths and released D12) emitted and independently decoded **24 Kodak images × five QPs = 480 complete bitstreams**. Payload bytes, full container bytes, stream SHA-256, per-stage entropy symbols/indexes, reconstructed latent and image equality are recorded for every case. [`analysis.json`](depth_bitstream/results/kodak_final_verified/analysis.json), [`case records and streams`](depth_bitstream/results/kodak_final_verified/) and [`RD/BD-rate figure`](depth_bitstream/results/kodak_final_verified/actual_byte_rd.pdf) are in Git.

| Codec | YUV 6:1:1 payload BD-rate vs released D12 | 95% paired-image bootstrap interval | RGB payload BD-rate | Full research-container YUV BD-rate |
|---|---:|---:|---:|---:|
| D2 | +7.980% | +7.095% to +8.826% | +7.428% | +7.825% |
| D4 | +4.512% | +4.019% to +4.995% | +4.100% | +4.420% |
| D6 | +2.758% | +2.368% to +3.128% | +2.307% | +2.703% |

BD-rate is calculated per image using PCHIP integration of log **emitted-byte** rate over the four-model common PSNR support, then averaged over the 24 images. Positive values mean extra bytes at matched quality. The rates use actual rANS payloads in the pinned CPU FP32 `FUFREF2` research wire format; the extra 88-byte integrity header is shown separately as sensitivity. **This is real decodable bitstream rate, but not Microsoft's native CUDA wire format or its deployment latency.** The model checkpoint files are identified by SHA-256 in [`artifact_manifest.json`](artifact_manifest.json) and currently reside on this server; distributing them is a remaining requirement for independent public replay.
