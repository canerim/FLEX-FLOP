# D2/D4/D6 actual-bitstream Kodak RD

`evaluate.py` encodes the 24 full-resolution Kodak RGB images at QP 0/16/32/48/63 with final-epoch D2/D4/D6 and Microsoft released D12. It writes real rANS payloads in the pinned **FUFREF2 CPU FP32 research format**, and asks an isolated decoder process to reconstruct from bytes alone. It checks exact reconstruction, latent hashes and each entropy stage's symbols and scale indexes. The rate used for BD-rate is `8 × emitted_payload_bytes / source_pixels`, not `bits_y + bits_z`. Both payload and research-container bytes are saved. The 88-byte integrity header is deliberately excluded from the main BD-rate. The format is **not Microsoft CUDA wire-compatible**, and CPU encode/decode time is **not deployment latency**.

`analyze.py` will refuse to output a BD-rate until all 480 cases and their isolated-decode checks have finished. It integrates PCHIP log-rate curves per Kodak image on the four-model common PSNR interval and averages 24 image percentages. It reports both RGB and YUV 6:1:1 PSNR with a paired-image bootstrap interval. Missing support or non-monotonic curves cause a failure instead of extrapolation.

```bash
cd /home/can_karsal/FLEX-PLUS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python \
  proof/depth_bitstream/evaluate.py
/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python \
  proof/depth_bitstream/analyze.py
```

The evaluator is resumable and records source, checkpoint, code and stream hashes. The `--max-cases` option is only a smoke test and marks the run partial. A native-CUDA-format RD comparison would be a separate protocol because the native and CPU research coders have different wire and numeric conventions.
