# Shared-exit CPU research-stream correctness preflight

Declared before execution, 28 September 2026. This is an engineering check, not a new RD comparison, native CUDA format, held-out router evaluation or timing benchmark.

- Fixed sources: the first frames of BasketballPass (416×240) and BQMall (832×480), the same two sources used in the earlier metric audit; no source is selected by the new outcomes.
- QP32 only. Replicate the source to multiples of256 to match the frozen shared-replay contract. Keep valid dimensions in the outer container.
- Four maps per source: previously frozen mean-router map, previously frozen Q90-router map, uniform deepest exit, and row-major cyclic6/8/10/12. No source-quality selection, fallback or map tuning.
- Reuse the pinned CPU FP32 FUFREF1 entropy implementation and isolated rANS binary. Its inner payload uses the released analysis/entropy weights, after checking all255 shared tensors against e15. An outer versioned container identifies the e15 reconstruction checkpoint and carries two-bit reachable-exit indices plus valid dimensions. All bytes, including redundant research identity headers, are accounted separately. This is not a production/minimum signalling format.
- A separate decoder process accepts only stream/output paths and resident checkpoint identities. Source analysis and hyperanalysis functions are replaced by functions that raise. Entropy symbols, indexes, recovered latent and final cropped tensors must match encoder-side reconstruction exactly for every case. CPU-only; two torch threads, no GPU.
- Verify the four maps share identical inner entropy bytes for each source. The stream decoder reads the map from the container; it receives no source pixels, source-error table, cached latent or encoder-generated entropy indexes.
- Exercise malformed-stream rejection (truncation, checksum, checkpoint identity and dimensions/count consistency). This is bounded input validation, not a security audit or cross-device determinism claim.
- Require all8 cases and every exact check before reporting completion. Stop on failure and retain evidence. Do not change active training, existing reference implementations or the paper's recorded reconstruction results.

The encoder sends a previously chosen map. This closes a small CPU decoder-input correctness check; it does not make the source-calibrated policy autonomous, measure map coding efficiency, or validate the full53-frame cohort in native CUDA.
