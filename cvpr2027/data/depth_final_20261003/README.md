# Final D2/D4/D6 Kodak comparison

`d6_epoch105_kodak.json` is an unmodified copy of the completed D6 run's
`kodak_final.json`. D2/D4 and Microsoft's released D12 inputs remain in
`../depth_final_20261002/`. Run
`python3 scripts/depth_final_three_20261003.py` from the paper root to
reproduce `analysis.json` and the vector figure without GPU access.

All four inputs contain 24 full-resolution Kodak images at QPs 0, 16, 32,
48 and 63. The BD-rate comparison integrates per-image monotone PCHIP
log-estimated-bpp curves on the **four-model common PSNR interval** before
averaging 24 percentages. Intervals are 5,000 image-paired bootstrap draws.
The 0.2-bpp PSNR contrast uses per-image interpolation with no
extrapolation. These rates are deterministic entropy estimates, not actual
bitstream/container bytes; the released model has a different training
history from the three scratch-trained models. `analysis.json` stores input
SHA-256 values and per-image BD-rate outputs.
