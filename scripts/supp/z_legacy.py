"""The old paper/supplementary.tex, moved here whole so that nothing is lost.

This is a transcription, not a rewrite. When `supplementary.tex` was cut down
to a shell of `\\input` lines, its six sections had to go somewhere, and this is
where they went. The prose is the prose that was there, converted from LaTeX to
the toolkit calls the other sections use. `paper/supp/z_legacy.tex` holds the
same content as LaTeX.

One thing did change, and it is recorded here so that nobody later reads an
edit as the original. The vocabulary follows the vocabulary `main.tex` now
uses: deblocking filter for seam-repair module, halo exchange for canvas
coupling, ring for band, exit for rung, cost multiplier for tilt, and affected
for contaminated. No number, no table row and no claim was altered.

It is unaudited, and it is meant to be temporary. A separate audit of this text
against `results/` and `paper/tables/macros.tex` found a number of statements
that do not match the files they claim to come from, several figures that no
file in `results/` contains at all, and several tables that mix checkpoints
without saying so. None of that is corrected here, because correcting it inside
a transcription would make it impossible to tell what was moved from what was
changed. A later agent folds each paragraph into the section that now owns the
subject, fixes it against the file, and deletes this module and its .tex
sibling.

Until then it sits at the end of ORDER, which is what the `z_` prefix is for.
"""


def tt(s):
    """Code and file names, in the monospace face LaTeX's \\texttt would give."""
    return f'<font face="Courier">{s}</font>'


def content(k):
    k.h1("Architecture details")

    k.par(r"<b>Costs.</b> With K exits over N = 12 trunk blocks and split "
          r"depth j, the cost of exit k in units of one released decode is")
    k.eq(r"c_k = s_{\mathrm{up}} + s_{\mathrm{trunk}}\frac{(k+1)b}{N} "
         r"+ s_{\mathrm{head}} + a_k + r")
    k.par(r"where b = N/K blocks per exit, s_up = 0.0816, s_trunk = 0.8944, "
          r"s_head = 0.0240 are the measured shares of the released decoder, "
          r"a_k is the adapter's share and r the deblocking filter's 0.0095. "
          r"Only the trunk term depends on k: the stem and head run for every "
          r"tile at every depth, which is why the ceiling 100(1 − c_j) is well "
          r"below 100 however shallow the ladder gets.")

    k.par(r"<b>Adapters.</b> Ad(f) = f + Wf, with W a C×C matrix and "
          r"zero-initialised, costs C² MAC/px, one eighth of a block's "
          r"8C² + 9C. The FFN variant, "
          r"Ad(f) = f + PW(WSiLUChunkAdd(PW(C→4C)(f))), costs 2C². Assignment "
          r"is by the number of blocks the exit skips: FFN at four or more, "
          r"the 1×1 otherwise.")

    # ------------------------------------------------------------------
    k.h1("The tiling penalty in detail")

    k.par(r"<b>Why only the depthwise matters.</b> A " + tt("DepthConvBlock")
          + r" at C = 384 costs 8C² + 9C = 1,183,104 MAC/px, of which the 3×3 "
            r"depthwise is 9C = 3,456, or 0.29%. Every other operator is 1×1 "
            r"and has no neighbours to miss. This is why the exact remedy is "
            r"affordable: haloing only the depthwise costs "
            r"+0.032% of the decode at 256 px tiles, against 1.745× for a halo "
            r"carried through the whole block. An earlier iteration of this "
            r"work measured that 1.745× correctly and then rejected haloing "
            r"altogether, instead of narrowing it to the one operator that "
            r"needs it.")

    k.par(r"<b>Seam penalty against per-tile depth.</b> Sweeping the split "
          r"depth j sweeps the number of per-tile blocks b = (K − j) b_exit "
          r"from 12 to 0, and the pure seam penalty follows.")
    k.rows(
        [["b", "area 1-((F-2b)/F)²", "q0", "q32", "q63"],
         ["12", "0.938", "0.175", "0.314", "0.606"],
         ["10", "0.859", "0.100", "0.149", "0.234"],
         ["8", "0.750", "0.065", "0.107", "0.194"],
         ["6", "0.609", "0.032", "0.053", "0.089"],
         ["4", "0.438", "0.009", "0.018", "0.038"],
         ["2", "0.234", "0.003", "0.006", "0.017"],
         ["0", "0.000", "<b>0.000</b>", "<b>0.000</b>", "<b>0.000</b>"]],
        r"Pure seam penalty in dB against the number of per-tile blocks b, "
        r"with every tile at full depth, over 12 sequences. The b = 0 row is "
        r"the control.")
    k.note(r"Transcribed from the old supplementary.tex, which described the "
           r"decoder as an untrained warm start and named no checkpoint. Not "
           r"yet reconciled against results/seam_vs_split.json.")

    k.par(r"The b = 0 row is the control and it is exactly zero at all three "
          r"rates, so the measurement is of the seam and nothing else.")

    k.par(r"The area fraction is a poor predictor: fitted with one free scale "
          r"it is wrong by 160--264% on average. It counts whether a pixel is "
          r"<i>reached</i> by the border and not how far the border propagates "
          r"into it, and it saturates: at b = 12 almost every pixel is already "
          r"reached, yet the penalty keeps rising steeply. A power law "
          r"fits well at every rate,")
    k.eq(r"\mathrm{seam} \propto b^{\alpha}, \qquad \alpha = 2.38,\ 2.22,\ "
         r"1.93 \ \mathrm{at}\ q0,\ q32,\ q63")
    k.par(r"with r = 0.98--0.995 in log--log and 12--22% mean relative error. "
          r"The exponent near two has a reading: the number of affected "
          r"pixels grows like the perimeter times the depth, proportional to "
          r"b, and the error accumulated in each of them grows with the number "
          r"of convolutions that reached it, also proportional to b. Once "
          r"the border has reached every pixel the first factor is fixed and only the "
          r"second keeps growing, which is the regime in which the area "
          r"fraction flattens and the measured penalty does not.")

    k.par(r"<b>The deblocking filter's ceiling.</b> Splitting per-pixel squared "
          r"error by distance from the nearest tile boundary at q63, with "
          r"the filter on against off and the same exit map:")
    k.rows(
        [["Ring (px)", "Share", "OFF", "Change"],
         ["0--4", "6.2%", "4.7×10<super>-5</super>", "−0.27%"],
         ["4--16", "17.3%", "4.4×10<super>-5</super>", "+0.05%"],
         ["16--64", "51.6%", "4.3×10<super>-5</super>", "+0.04%"],
         ["64--128", "25.0%", "4.1×10<super>-5</super>", "+0.04%"]],
        r"Per-pixel squared error by distance from the nearest tile boundary "
        r"at q63, filter off, and the change when it is switched on.")
    k.note(r"Transcribed from the old supplementary.tex, which named no "
           r"checkpoint. Not yet reconciled against "
           r"results/seam_spatial_published.json, whose own provenance field "
           r"says the filter-on column cannot be reproduced from anything now "
           r"on disk.")

    k.par(r"The error is 15% higher in the boundary ring than in the interior "
          r"and falls monotonically with distance, so the artefact is real and "
          r"local. But the gate leaks: the filter gains only in the first "
          r"ring. Taking the boundary gain and assuming a perfect gate "
          r"elsewhere bounds what it could ever earn at "
          r"0.062 × 4.7×10<super>-5</super> × 0.0027 / 4.30×10<super>-5</super> ≈ 1.8×10<super>-4</super> of the frame's "
          r"error, that is, about 0.0008 dB, for 0.95% of the decode.")

    # ------------------------------------------------------------------
    k.h1("Reproducibility")

    k.par(r"<b>Frozen encoder.</b> Every evaluation asserts "
          r"max|θ_ours^enc − θ_released^enc| = 0 before measuring. Without it "
          r"the comparison is not on the same latent and the claim of an "
          r"unchanged bitstream is unverified.")

    k.par(r"<b>Warm-start equivalence.</b> At initialisation the deepest exit "
          r"reproduces the released decoder bit-exactly at q0/32/63, because "
          r"every inherited weight is copied and every new module's last layer "
          r"is zero.")

    k.par(r"<b>Sorted execution.</b> The reordering of Section 5 is "
          r"bit-identical on CUDA under a random exit map and random content. "
          r"It is <i>not</i> bit-identical on CPU, and the reason is not the "
          r"algorithm: a plain " + tt("DepthConvBlock") + r" is already "
          r"order-dependent at batch size 13 there, by about 10<super>-7</super>, while exact "
          r"at 4, 9, 16 and 40, because oneDNN selects a different blocking. A "
          r"test that compared the two paths on a decode of an all-zero image "
          r"would pass vacuously (every tile carries the same latent, so any "
          r"permutation is trivially exact), and ours therefore uses random "
          r"content.")

    k.par(r"<b>Two decibel conventions.</b> Pooling every tile of every frame "
          r"into one MSE and averaging a per-frame decibel differ by "
          r"0.023--0.033 dB on an identical allocation, a quarter to a third "
          r"of the working budget, with pooling always the flattering one. We "
          r"report per-frame, which is what the reference implementation of "
          r"the baseline computes.")

    k.par(r"<b>Two quality metrics.</b> An RGB MSE ratio and the "
          r"6:1:1-weighted YUV 4:2:0 PSNR the baseline reports agree within "
          r"0.003 dB on identical decodes across the whole rate range; we "
          r"verified this rather than assuming it.")

    k.par(r"<b>Timing.</b> " + tt("torch.cuda.Event") + r" is created on the "
          r"process's current device, so a timing harness on a non-default GPU "
          r"measures nothing in particular and says nothing about it. Every "
          r"wall-clock figure here was re-measured with the device pinned, and "
          r"a test asserts that every script using CUDA events pins it.")

    k.par(r"<b>Generated numbers.</b> No number in the paper is typed. Every "
          r"table and every inline figure is generated from " + tt("results/")
          + r" by one script, expanded as a macro, and a second script "
            r"re-reads \NumClaims of them out of the prose and compares them "
            r"against the same files, exiting non-zero on any disagreement. It "
            r"has caught four drifts, one of which was a figure count that "
            r"changed when a figure was inserted in the middle.")

    # ------------------------------------------------------------------
    k.h1("Claims this work made and then measured")

    k.par(r"Five statements that appeared in earlier drafts of this work, and "
          r"what the experiment said. Each was plausible and argued from a "
          r"mechanism, and in every case it took a measurement rather than a "
          r"better argument to settle it.")
    k.bullets([
        r"<b>“The corrupted-area fraction explains why split depth "
        r"matters.”</b> It predicts the seam with 160--264% error, because the "
        r"affected area saturates once the border reaches every pixel and the "
        r"damage does "
        r"not. What replaces it is a power law in b with the exponent "
        r"<i>fitted</i> (2.38, 2.22, 1.93 at q0, q32, q63), which fits to "
        r"12--22%. A fixed square, which earlier drafts quoted as though it "
        r"were the same thing, fits to 31--38%: better than the area by a "
        r"factor of five and worse than the fitted exponent by a factor of "
        r"two.",

        r"<b>“The halo exchange is the clearest remaining gain, worth 3.5--5 "
        r"points.”</b> It removes 47--91% of the floor and collapses the "
        r"routed saving: \CoupPaddedMid% to \CoupCoupledMid% at q32, "
        r"\CoupPaddedHigh% to \CoupCoupledHigh% at q63. The exchange is exact at "
        r"uniform depth, and routing is the deliberate violation of that "
        r"condition.",

        r"<b>“A tile size chosen per resolution is the obvious "
        r"extension.”</b> 128 px tiles help only at 416×240 (+2.4 points) and "
        r"cost 1.7--3.3 points elsewhere, including at 832×480 where the tile "
        r"count rises from 8 to 28 and the saving still falls.",

        r"<b>“Configuration A costs the encoder about 1.21 decodes.”</b> The "
        r"exact search costs 4.6. The 1.21 figure was the full-frame table, "
        r"which may be used to rank but not to report.",

        r"<b>“The recovery curve of partial signalling <i>is</i> the Lorenz "
        r"curve of the per-tile regret.”</b> It is neither an identity nor, as "
        r"the second version of this claim said, an upper bound. Overriding a "
        r"set removes exactly its regret at a <i>fixed</i> multiplier, but the "
        r"constraint is distortion, and returning to the budget re-bisects λ. "
        r"What the re-bisection opens up is a larger feasible set: the hybrid "
        r"runs the oracle's λ on the overridden tiles and the router's fixed β "
        r"on the rest, and two multipliers reach allocations one cannot. "
        r"Measured, the recovery exceeds the Lorenz prediction wherever that "
        r"matters, reaching \LorenzTightMin--\LorenzTightMax% of it across the "
        r"sweep, and at \HybridBeatsAN of \HybridBeatsAOf rates a half map "
        r"beats a complete one outright. Both versions of the claim took a "
        r"separability argument that holds for the objective and applied it to "
        r"a saving measured at fixed distortion.",
    ])

    # ------------------------------------------------------------------
    k.h1("Negative results")

    k.par(r"We list these because they cost real time and are not otherwise "
          r"recoverable from the paper.")
    k.bullets([
        r"<b>First-order border extrapolation.</b> Continuing the local "
        r"gradient past a tile boundary is 1.8× worse than replication, a "
        r"zero-order hold, "
        r"at q63 (\SeamLinearHigh dB against \SeamReplHigh). Amplifying "
        r"boundary noise costs more than the trend recovers.",

        r"<b>Fitted AR(1) padding</b> [11]. The best estimator we measured, "
        r"and rejected: 0.02 dB for 10.7% of decode wall-clock.",

        r"<b>A learned deblocking filter.</b> Bounded above at about 0.0008 dB for "
        r"0.95% of decode, even with a perfect gate.",

        r"<b>Jointly training the router with the decoder.</b> In one "
        r"configuration the router collapsed to a rate-dependent constant with "
        r"zero agreement with the oracle; in another it did not. Training the "
        r"router against a frozen decoder is reliable and is what we report.",

        r"<b>A halo through the per-tile trunk.</b> 1.745× the cost of a full "
        r"decode, which is more than routing saves.",

        r"<b>A rank-1 surrogate for the per-tile distortion.</b> Modelling "
        r"log D(t,k) ≈ α log b(t) + c + log φ_k from the tile's bit count "
        r"gives a routing rule with no learned parameters, and it works, but "
        r"only because it places tiles on the ladder by <i>level</i>. What "
        r"decides a tile is the <i>spread</i> across the ladder, which a "
        r"rank-1 model in the level cannot represent, and the two are not the "
        r"same ordering: bits correlate with distortion at every exit and with "
        r"the depth the oracle assigns, at ρ ≈ 0.53, but a level-only model "
        r"cannot see which tiles have a steep ladder and which have a flat "
        r"one.",

        r"<b>A single router across all rates.</b> It comes within \GapMin "
        r"points of the oracle at q\GapMinQp, where the cost multiplier the "
        r"bisection has to apply is essentially zero because that is where its training λ "
        r"lands, and falls to \GapMax at the far end. The gap tracks |β| in "
        r"both directions. The remedy is a router per operating point.",
    ])

    # ------------------------------------------------------------------
    k.h1("Things that were not running")

    k.par(r"Four checks this work cites were not measuring what they claimed, "
          r"and all four were found by running them rather than by reading "
          r"them.")
    k.bullets([
        r"The bit-exactness test for the warm start declared " + tt("qp")
        + r" as a bare parameter with no " + tt("parametrize") + r", so "
        + tt("pytest") + r" looked for a fixture of that name and errored at "
        r"setup. It has been cited by name for the claim that the deepest exit "
        r"reproduces the released decoder exactly. Parametrised over three "
        r"rates and every warm start on disk: six cases, all exact.",

        r"The equality check for the sorted per-tile loop compared the two "
        r"paths on a decode of " + tt("torch.zeros") + r", where every tile "
        r"carries the same latent and any permutation is trivially exact. On "
        r"random content with a mixed exit map the result is still "
        r"bit-identical on CUDA, but the original check could not have failed.",

        r"Every wall-clock measurement in this work was taken with "
        + tt("torch.cuda.Event") + r", which is created on the process's "
        r"<i>current</i> device rather than on the device the tensors live on. "
        r"The timing runs used a non-default GPU, so the events and the work "
        r"were on different cards. This does not raise: it returns numbers. "
        r"The tell, in hindsight, was that the released decoder appeared to "
        r"take 274, 401 and 512 ms at q0, q32 and q63; a fixed-architecture "
        r"synthesis network whose arithmetic does not depend on the quality "
        r"index cannot get slower with it. Re-measured with the device pinned "
        r"it is 111 ms at all three. The corrections are not small and one of "
        r"them reverses a claim: the tiling overhead falls from 8.5% to "
        r"\TilingOverhead%, and the unsorted per-tile loop, reported as "
        r"<i>slower</i> than the dense decoder at high rate, in fact realises "
        r"\WallMasked% at q0. Sorting the tiles is therefore a "
        r"\WallSortedGain-point improvement on a loop that already worked. The "
        r"wall-clock section of the main paper is written against the "
        r"corrected numbers. Anything in this repository that timed a "
        r"non-default GPU should be assumed to have had this bug; a source "
        r"test now asserts that every script using " + tt("torch.cuda.Event")
        + r" pins the device first, and it found two more.",

        r"The router head suppresses exits below the split depth by assigning "
        r"them −10<super>4</super>. That is a mask only while the head's own logits stay well "
        r"above it, and this head's did not: nothing in a cross-entropy or a "
        r"regret objective penalises a common offset, one drifted in during "
        r"training, and the raw outputs settled near −10<super>4</super>. The two suppressed "
        r"entries then became the <i>largest</i> in every row, took nearly all "
        r"the probability mass, and the caller's clamp turned the resulting "
        r"choice into the cheapest real exit, so a large share of every "
        r"configuration-B allocation went to the cheapest exit for a reason "
        r"unrelated to the tile. The mask is now −∞, which cannot drift, and a "
        r"test checks the property across six logit scales rather than "
        r"checking the constant.",
    ])
