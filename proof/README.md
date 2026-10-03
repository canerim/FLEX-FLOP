# Public evidence workspace

This directory separates two different experiments. The shared-latent **e15 early-exit** decoder is compared with Microsoft's released DCVC-UF-Intra D12 in [`early_exit_vs_released`](early_exit_vs_released/README.md). Independently trained **D2/D4/D6** codecs are assessed for actual bitstream BD-rate in [`depth_bitstream`](depth_bitstream/README.md). A speedup measured for one system must not be attributed to the other.

Only measurements with raw paired samples, exact source/checkpoint identities, quality measurements, and an otherwise idle GPU qualify as speedup evidence. Existing 2.4–2.6× Triton measurements compare e15 variants with one another on a shared GPU; they are **not** released-D12 speedups. Existing D2/D4/D6 Kodak BD-rate figures use entropy estimates; they are **not** actual-byte BD-rates.
