# Literature check: adaptive synthesis and sparse boundary execution

This memo uses primary conference/arXiv papers inspected on 4 October 2026.
It informs positioning and the next experiment; it is not evidence that the
current e15 decoder is faster end to end.

1. [AdaNIC (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.html) already routes spatial transform capacity in a learned image codec and reports speed/BD-rate tradeoffs. Our paper cannot claim to invent spatially adaptive codec compute. Its distinct evaluated axis is fixed-latent **synthesis depth** after a common DCVC-UF analysis/entropy path, with true bitstreams and separately audited tile-context effects.
2. [ClassSR (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf), [APE (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf), and [ENAF (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Nguyen_ENAF_A_Multi-Exit_Network_with_an_Adaptive_Patch_Fusion_for_WACV_2025_paper.html) establish patch selection, shared early exits, and quality-aware patch fusion in super-resolution. ENAF explicitly estimates patch reconstruction quality for routing. Our map-only risk predictor's weak DIV2K transfer reinforces that a decoder-available quality estimate is more promising than exit histograms, but we have **not** measured such an estimator in codec inference. ENAF belongs in related work; it has now been cited there.
3. [Dynamic Convolutions (CVPR 2020)](https://arxiv.org/abs/1912.03203) implements spatially sparse execution via gather/scatter and reports wall-clock speedup, not only FLOPs. It is a concrete systems baseline for a future exact-context dependency-band scheduler. Our current union-of-required-cells estimate is an arithmetic idealization; its 0.965× cost on `0828/QP63` leaves too little margin to assume any practical acceleration after sparse packing and launches.
4. [What Matters in Practical Learned Image Compression (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html) selects codecs under a device runtime target and, for on-device comparisons, deploys baselines with matched optimizations. The relevant standard for our paper is likewise matched released/e15 backends and bitstream-to-image latency at each QP. Analytical synthesis MAC savings and accelerated synthesis-stage kernels remain separate quantities.

**Decision:** prioritize a decoder-available quality-risk signal with fresh
training/validation images and a strong cost-aware fallback comparator. For
boundary handling, prototype an exact dependency-band scheduler only if its
measured end-to-end gain survives packing and kernel overhead; QP63's ideal
3.5% MAC margin makes that particular map a poor first target. Keep AdaNIC,
ClassSR/APE/ENAF, and spatial-sparse kernels distinct in related work rather
than implying the codec result is the first instance of any of them.
