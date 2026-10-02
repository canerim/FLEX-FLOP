# Introduction revision, 2026-10-02

The current title is **RegLIC: Region-Adaptive Learned Image Compression**.
The opening follows the problem–mechanism–evidence pattern visible in
the introductions of the [2025 CVPR Best Paper, VGGT](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)
and [2025 Best Student Paper, Neural Inverse Rendering from Propagating
Light](https://openaccess.thecvf.com/content/CVPR2025/papers/Malik_Neural_Inverse_Rendering_from_Propagating_Light_CVPR_2025_paper.pdf).
Their concrete motivation and early account of the contribution inform
the structure; no phrasing or figure design was copied.

Closest technical precedents are
[ClassSR](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)
(patch-to-network assignment),
[APE](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf)
(adaptive patch exiting in restoration),
[AdaNIC](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.pdf)
(spatial transform routing in a codec), and the
[DCVC-UF implementation](https://github.com/microsoft/DCVC)
(the actual codec studied). The introduction therefore avoids claiming
that patch routing, early exit, or dynamic compression was invented here.
It identifies the testable distinction: conditional synthesis depth
with one coded latent, executed tile exits and a shared output path.

The model bank is described as a complementary study, with no reported
six-way router result. The introduction illustration carries no numerical result. The reported
27.97% synthesis-MAC saving comes from a source-calibrated model; the
text separates it from unresolved fixed-control routing and unmeasured
end-to-end latency.
