# FLEX — abstract (main paper)

We introduce FLEX, a method that adaptively adjusts the decoding cost of a
learned image codec to the content of each region it reconstructs. FLEX
achieves this by letting different parts of a frame leave the decoder's
residual trunk at different depths, so that flat regions are decoded by a
fraction of the network while textured ones run it in full. We reformulate
exit selection as a constrained allocation problem: the distortion of every
exit is tabulated, the compute of every exit is closed form, and their
Lagrangian combination admits a per-region minimiser that is exact rather than
heuristic. The fully convolutional structure of the decoder is what makes the
mechanism free — depth is varied without modifying the architecture, the
entropy model, the bitstream syntax, or the inference hardware. We demonstrate
that FLEX requires no side information at all: a 144k-parameter head costing
0.163% of one decode recovers the allocation from tensors the decode has
already produced, leaving the file byte for byte the released one. We further
introduce zero-initialised residual adapters, which keep the deepest exit
bit-exact with the stock decoder and so bound the cost of adaptivity at full
depth by construction. On the standard test conditions, FLEX removes 27.6% of
decoder multiply-accumulates on average and 33.7% at the lowest rate for a
0.1 dB drop, at a signalled cost of 64 to 91 bits per frame, four orders of
magnitude below the frame it steers — and keeps 24.7% with no added bits.
Measured kernel latency follows the retained fraction to within 0.036, and at
its saturation budget of 0.121 dB the lowest rate reaches the ceiling of the
ladder, 40.1%.
