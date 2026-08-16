"""The knobs of FLEX-UF, in one place, with the reason each default is what it is.

Every value is either (a) read off the DCVC-UF source, (b) measured by
`scripts/mac_audit.py`, or (c) taken from a paper we can cite. Nothing is a guess.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fixed by the DCVC-UF architecture — NOT tunable. Read from
# ~/DCVC/src/models/image_model.py:15-18, verified by a live forward pass.
# ---------------------------------------------------------------------------
LATENT_CH = 256      # g_ch_y       — channels of y_hat entering the decoder
TRUNK_CH = 384       # g_ch_enc_dec — width of every trunk DepthConvBlock
PRESHUFFLE_CH = 192  # g_ch_src = 3*8*8 — head output, before PixelShuffle(8)
N_TRUNK_BLOCKS = 12  # dec_1[1..12]; dec_1[0] is the opening ResidualBlockUpsample
UPSAMPLE_FACTOR = 2  # the opening block's SubpelConv2x
SHUFFLE_FACTOR = 8   # F.pixel_shuffle(out, 8) at the very end
PIXELS_PER_LATENT = UPSAMPLE_FACTOR * SHUFFLE_FACTOR  # 16
QP_LEVELS = 64       # CompressionModel.qp_num(); confirmed live on DMCI

# Microsoft's own recipe constants (train_image.py:22-32, train_image.sh).
RECIPE_BATCH = 16
RECIPE_LAMBDAS = (10.0, 2048.0)   # log-spaced across the 64 QPs
RECIPE_GRAD_CLIP = 0.1
RECIPE_EPOCHS = 105


@dataclass
class FlexUFConfig:
    """One experiment's configuration."""

    # -- the exit ladder ----------------------------------------------------
    num_exits: int = 6
    """K — the number of exits the 12 trunk blocks are grouped into (b = 12/K = 2).

    Why 6: FLEX-FLOP swept this and found it is not a free knob — K=3 and K=12
    both collapse, with the router putting >=97% of patches on one or two exits.
    K=6 was its operating point, and UF has the identical 12-block trunk.
    """

    split_depth: int = 2
    """j — how many exit-groups run FULL-FRAME before the decode goes per-patch.

    This is the "ilk birkaç blokta ortak ilerlet, sonra patch'lere böl" structure,
    and it is also ClassSR's: a shared stem, then per-sub-image branches of
    different capacity.

    Why j=2 by default: groups 0..1 = 4 of 12 trunk blocks run once for the whole
    frame (no seams), leaving 8 blocks routable. The measured cost table says the
    routable region is where all the saving lives, and j=2 leaves the most of it:
    FLEX got 23% saving at j=2 versus 15% at j=4. j is the axis experiment E2
    isolates.

    j = K is legal and is the **correctness control**: with every group full-frame
    the hybrid path must reproduce the plain full decode bit-exactly.
    """

    latent_patch: int = 8
    """p — patch side in LATENT pixels. p=8 -> 16x16 feature tile -> 128x128 RGB.

    Why 128x128 and not 64x64:
      1. The Boundary Law (FLEX, R^2=0.9995) says the excess MSE of patch-wise
         decoding scales with *seam density*, i.e. as 1/patch-side. Measured
         penalties: 0.856 dB at 128 px, 2.484 dB at 64 px — 128 costs about a
         third as much distortion for the same routing idea.
      2. Microsoft trains at 256x256 crops for the first 90 epochs and 512x512
         after (train_image.py:22-32). 128 divides both cleanly: 2x2 = 4 tiles
         per 256-crop, 4x4 = 16 per 512-crop, and at 1080p inference it gives
         15x8 = 120 tiles — still plenty of routing granularity.
      3. A larger tile also amortises the halo: the halo overhead is
         ((F+2h)/F)^2, so at F=16 a 2-latent-px halo costs 1.56x on the halo'd
         stage, versus 2.25x at F=8.
    Experiment E3 sweeps back down to 64 to measure the trade directly.
    """

    latent_halo: int = 2
    """h — context padded around each tile, measured in LATENT pixels.

    Deliberately specified in latent units, not feature units, and deliberately
    larger than 1:
      - The per-patch region runs (K-j)*b trunk blocks, each with a 3x3 depthwise.
        At j=2, K=6, b=2 that is 8 blocks, so a tile's output legitimately depends
        on 8 feature pixels beyond its border. A 1-pixel halo cannot cover that.
      - h latent px = 2h feature px of real context (the opening block upsamples
        by 2), so h=2 gives 4 feature px — half the true receptive field, and the
        seam damage falls off sharply with distance.
      - Undersizing the halo hurts the *router* most, not the PSNR average: the
        router sees per-tile distortions contaminated by border artefacts and
        learns to route on seam noise instead of on content difficulty.
    Cost: only the halo'd stage pays ((16+2*4)/16)^2 = 2.25x, and only the head
    is halo'd when `full_frame_head` is False.
    """

    trunk_halo: int = 0
    """Halo carried through the PER-TILE TRUNK, in latent px. Default 0 = none.

    This is separate from `latent_halo` (which the router reads) on purpose,
    because the two have wildly different prices.

    A haloed tile computes ((F + 2h) / F)^2 times the pixels it keeps. At F=16
    feature px and h=2 latent px (4 feature px) that is 2.25x — and applied to
    the per-tile trunk, which is the expensive part, it does not just erode the
    saving, it inverts it. Measured directly by
    `tests/test_cost_matches_reality.py`: with the trunk halo on, decoding every
    tile at the DEEPEST exit cost **1.745x a plain full decode**. Paying 74%
    extra to save nothing.

    The seam is dealt with where it is affordable instead: the head runs
    full-frame over the stitched canvas (FLEX: the head carries ~75% of the
    patch-independence penalty), and the exit adapters — being 1x1, with no
    receptive field — add none of their own.

    Set > 0 only to measure what the trunk halo buys, never as a default.
    """

    tile_pad_mode: str = "arls"
    """How a tile's border is padded inside the PER-TILE trunk blocks.

    Free, and the largest single quality win measured on this project.

    The only spatial operator in a DepthConvBlock is one 3x3 depthwise, and under
    a j-split it meets the tile border with nothing beyond it. Stock padding is
    zeros — which is a terrible estimate of a neighbour that is, in reality,
    usually close to the edge value. Replicating the edge is nearly free to
    compute and far closer to the truth.

    Measured on the released weights, pure seam cost with every tile at the
    deepest exit (so no early exit at all), always against the STOCK full-frame
    decode as reference:

        qp        zeros      replicate     gain
         0       0.1306        0.1333     -0.003 dB
        32       0.3521        0.2960     +0.056 dB
        63       0.8727        0.6315     +0.241 dB

    At qp63 that removes 28% of the seam for the same weights and the same FLOPs;
    at low rate it is within noise. Worth taking because the cost is exactly zero
    — it is a string on a module, consumed by the same kernel.

    A first measurement of this claimed +0.794 dB and was wrong: it applied
    replicate to EVERY block, which moved the reference full-frame decode too.
    Shifting both sides shrinks the apparent gap and inflates the gain. The
    reference must always be stock UF.
    It applies ONLY while decoding per-tile; full-frame decode keeps zero padding
    so `forward_full` stays bit-exact against stock UF and the control still means
    something.

    "zeros" reproduces the stock behaviour, for measuring what this is worth.
    """

    seam_repair: str = "full"
    """A cheap full-frame pass after stitching, to heal the tile borders.

    "none"       — off, the stock behaviour
    "depthwise"  — 3x3 depthwise only, ~0.025% of the decode
    "full"       — 3x3 depthwise + 1x1, ~1.09% of the decode

    Why here and not in the trunk: the seam is a spatial artefact and repairing
    it needs a 3x3 that can see ACROSS a tile boundary. Inside the per-tile trunk
    no kernel ever can — that is what the halo would have bought, at 2.25x on the
    expensive part, which inverted the saving entirely. After unpatchify the
    canvas is whole again, so one pass there gets the same reach for a fixed ~1%.

    Against a per-tile trunk block at 7.45% of the decode, "full" is an eighth of
    one block. Zero-initialised, so it is exactly the identity until trained and
    cannot make anything worse.
    """

    adapter_kind: str = "conv1x1"
    """Which adapter sits between an early exit and the shared head.

    "conv1x1"  — a single trainable 1x1 convolution, residual, zero-initialised.
    "ffn"      — the pointwise expand/activate/contract pair that mirrors the
                 DepthConvBlock FFN the early exit skipped.

    Why a 1x1 and not something with spatial extent: the capacity an early exit
    is missing is overwhelmingly *FFN* capacity. A DepthConvBlock costs
    8C^2 + 9C MAC/px, of which the FFN alone is 6C^2 = 74.8%; the only spatial
    operator in the entire block is one 3x3 depthwise worth 9C, i.e. 0.3%.
    Exiting early therefore removes almost purely pointwise channel-mixing
    capacity, and a pointwise adapter is the matched replacement.

    It also has a second, structural benefit: a 1x1 convolution has *no*
    receptive field, so it contributes exactly zero patch-boundary penalty. An
    adapter with a 3x3 in it would add seam damage precisely at the tiles that
    took an early exit — the ones already least able to afford it.
    """

    adapter_expand: int = 4
    """Expansion ratio for adapter_kind="ffn". 4 matches the DepthConvBlock FFN
    (Conv2d(C, 4C, 1) -> WSiLUChunkAdd -> Conv2d(C, C, 1))."""

    full_frame_head: bool = True
    """Run the head over the stitched canvas rather than per tile.

    Why default True: FLEX's most consequential training result. The head is
    2.40% of decoder MACs [measured] but carries ~75% of the patch-independence
    penalty, because its 3x3 depthwise convolves tile borders against zero
    padding and PixelShuffle(8) smears each border into an 8-px RGB band.
    Adapters trained *through* a full-frame head gained +0.51..+0.90 dB; the same
    head bolted onto per-patch-trained adapters LOST 0.14..0.24 dB. The sign
    flips. Train through the decode path you deploy.
    """

    # -- the joint multi-exit objective (Scardapane et al. 2020, Eq. 6-7) ----
    aux_weight: float = 1.0
    """alpha_i in Eq. (6) — the weight on each auxiliary (early) exit's loss.

    Eq. (6):  min  L + sum_i alpha_i * L_i
    Eq. (7):  L_i = sum_n l(y_n, c_i(x_n))

    Why 1.0: MSDNet swept per-exit weights and reported "using the same weight
    for all loss functions works well in practice". Equal weighting also keeps
    the deepest exit from being drowned out, which matters here because the
    deepest exit is the quality anchor the whole frontier is measured against.
    """

    aux_schedule: str = "constant"
    """How alpha_i evolves. "constant" | "warmup".

    "warmup" ramps alpha from 0 to aux_weight over the first `aux_warmup_epochs`,
    so the model first learns to be a good codec at full depth and only then is
    asked to also be good shallow. This guards against the FLEX from-scratch
    failure mode where every exit converges to the same mediocre point (~26 dB)
    with no differentiation.
    """

    aux_warmup_epochs: int = 10

    # -- rate ---------------------------------------------------------------
    qp: int = 63
    """Evaluation QP. Training samples all 64 uniformly, as Microsoft does
    (image_dataset.py:58). q63 is the hardest case for patching: the penalty
    grows with rate (1.02 dB at qp30 -> 2.48 dB at qp63)."""

    def __post_init__(self):
        if N_TRUNK_BLOCKS % self.num_exits != 0:
            raise ValueError(
                f"num_exits={self.num_exits} must divide {N_TRUNK_BLOCKS} trunk blocks"
            )
        if not 0 <= self.split_depth <= self.num_exits:
            raise ValueError(f"split_depth={self.split_depth} must satisfy 0 <= j <= K")
        if self.adapter_kind not in ("conv1x1", "ffn"):
            raise ValueError(f"unknown adapter_kind {self.adapter_kind!r}")

    @property
    def blocks_per_exit(self) -> int:
        """b = 12/K — trunk blocks in one exit group."""
        return N_TRUNK_BLOCKS // self.num_exits

    @property
    def feature_patch(self) -> int:
        """Tile side in FEATURE pixels (after the 2x opening upsample)."""
        return self.latent_patch * UPSAMPLE_FACTOR

    @property
    def feature_halo(self) -> int:
        """Halo in FEATURE pixels — what the code actually pads with."""
        return self.latent_halo * UPSAMPLE_FACTOR

    @property
    def rgb_patch(self) -> int:
        """Tile side in RGB pixels."""
        return self.latent_patch * PIXELS_PER_LATENT

    def tiles_for_crop(self, crop: int) -> int:
        """How many tiles a `crop` x `crop` training crop yields."""
        return (crop // self.rgb_patch) ** 2
