# Related-work claim audit

Checked against primary papers on 28 September 2026. The comparison concerns
mechanisms and experimental controls; no performance number from another
codec is inserted into the DCVC-UF results.

| Primary source | Relevant mechanism | Consequence for this manuscript |
|---|---|---|
| [ClassSR, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf), §3 | Learned patch assignment among differently sized SR networks | Attribute the patch-to-network principle. The independent codec bank additionally changes the representation and rate. |
| [APE, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf), §3.1–3.3 | Incremental-benefit prediction for patch exits; a common reconstruction tail | Neither marginal-benefit routing nor a shared tail alone distinguishes FLEX-UF. Test codec-specific exit alignment and assembled reconstruction. |
| [RBQE, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610273.pdf) | Quality-dependent early termination of compressed-image enhancement | Distinguish post-decoding enhancement from conditional synthesis inside the codec. |
| [Swift, NSDI 2022](https://www.usenix.org/system/files/nsdi22-paper-dasari.pdf), §4.1 | Compute-dependent exit heads in a layered neural video decoder, at different output resolutions | Early exit in a neural codec has a precedent. FLEX-UF allocates spatial depth at fixed output resolution. |
| [SlimCAE, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Slimmable_Compressive_Autoencoders_for_Practical_Neural_Image_Compression_CVPR_2021_paper.pdf) | Joint rate–distortion optimisation at several widths | Attribute flexible capacity; distinguish width from the nested depth axis used here. |
| [AdaNIC, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.pdf), §3.2–3.4 | Blockwise spatially varying transform widths and a learned routing agent | Closest spatial-capacity compression precedent. A “first adaptive compression” claim would be unsupported. |
| [cgSlimDecoder, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Complexity-Guided_Slimmable_Decoder_for_Efficient_Deep_Video_Compression_CVPR_2023_paper.pdf), §3 | Complexity-conditioned channel allocation across video-decoder modules | Distinguish spatial stopping from width allocation across motion/residual modules. |
| [Shallow Decoders, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.pdf), abstract and §3 | Globally shallow synthesis with stronger encoding | A shallow fixed-depth control is necessary; encoder work belongs in the accounting. |
| [DCVC-RT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf), §3 | Operational costs such as memory traffic and module calls limit speed | Avoid converting synthesis MAC percentages into measured codec latency. |
| [DCVC-UF, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Ultra-Fast_Neural_Video_Compression_CVPR_2026_paper.html) | Chunk-based video coding and streamlined entropy interaction | Cite the release accurately, but confine the experimental claim to the supplied intra model. |
| [Spatial Competition, accepted ICIP 2026](https://arxiv.org/html/2605.13243v1), §2.2–2.3 | Same-architecture specialised codecs; regional RD search; mode signalling; connected-region processing | Region merging is a controlled application of an existing strategy. The proposed bank varies synthesis depth and aims to reduce candidate search. Acceptance is confirmed on the authors' [arXiv record](https://arxiv.org/abs/2605.13243). |

## What is distinctive enough to test

The current mechanism is a shared DCVC-UF latent and full-frame prefix,
conditional constant-width block pairs, pointwise exit alignment and a
shared assembled reconstruction. Its scientific test is whether content
placement adds value after controlling average depth, actual quality,
reference choice and decision information. The ongoing codec bank tests
representation specialisation against reuse; it is not an extra set of
reachable exits in the present checkpoint.

The closest missing method controls are equal-budget retraining without
adapters/repair, an APE-inspired incremental-gain predictor under the same
calibration protocol, and a validation-fixed choice between blind and learned
allocation. These are proposed comparisons, not measured advantages over the
cited methods.

## Editorial principles used

Move from a concrete inefficiency to a constrained design decision, then to
an experiment that can falsify the intended benefit. Explain each baseline's
role before presenting its number. Separate completed evidence from the next
hypothesis. These organisational choices do not copy another paper's prose,
figures or contribution claims.
