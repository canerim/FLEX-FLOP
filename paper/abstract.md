# FLEX — abstract (main paper)

For learned image compression, decoding cost is fixed by the architecture:
every region of every frame is reconstructed by the same number of layers,
however little of the network it needs. We propose FLEX, which adaptively
adjusts decoding depth to the content of each region, letting flat areas leave
the decoder's residual trunk early while textured ones run it in full. We
formulate the choice of exit as a constrained allocation problem — tabulated
per-exit distortion against closed-form per-exit compute — whose per-region
minimiser is exact rather than heuristic. The mechanism is deterministic and
leaves the architecture, the entropy model and the bitstream syntax untouched,
so it applies without modification to files that were encoded before it
existed. We demonstrate that no side information is required: a 144k-parameter
head, 0.163% of one decode, recovers the allocation from tensors the decode has
already produced. We further introduce zero-initialised residual adapters,
which keep the deepest exit bit-exact with the stock decoder, so adaptivity
cannot cost quality at full depth. Removing 24.7% of decoder
multiply-accumulates with the file left byte for byte unchanged, and 27.6% on
average — 33.7% at the lowest rate — once the map is signalled at 64 to 91 bits
per frame, FLEX cuts decoding computation by a wide margin while the
rate-distortion curve barely moves, within 0.1 dB. Additionally, we find that
each rate exhausts the ladder at a tolerance of its own, from 0.121 dB at the
lowest to 0.270 dB at the highest, and that the lowest rate's ceiling is 40.1%.
To the best of our knowledge, this is the first content-adaptive depth
mechanism for a learned decoder that leaves the bitstream unchanged.
