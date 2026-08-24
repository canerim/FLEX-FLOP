"""A theory supplement: what changes, mathematically, when the cut is removed.

The main paper's theory.tex already covers the tiled ladder. Its first
assumption is cost additivity, and it says so plainly: it "holds whenever
tiles are decoded independently, which is what a tiled decoder does". This
document is about what happens when that stops being true -- because
per-position decoding replaces the tiling with a dilation, and the dilation
couples neighbouring cells.

Three things change and each is provable. The distortion becomes exactly
separable, which it was not before. The cost stops being additive and becomes
convex. And the split depth, which existed only to bound the seam, stops
having a role, which moves the floor and therefore the ceiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reportlab.lib.styles import ParagraphStyle
from build_report import (P, H1, H2, BODY, ABST, TITLE, AUTH, MONO, CAP, INK2,
                          display, theorem, proof, table, build, J, RES,
                          Spacer, STMT, THM, PRF)
import json


def C():
    return json.loads((RES / "cost_constants.json").read_text())


def story():
    c = C()
    u, h, tr = c["SHARE_UPSAMPLE"], c["SHARE_HEAD"], c["SHARE_TRUNK"]
    pb, bpe, K = c["per_block"], c["blocks_per_exit"], c["K"]
    ad = c["adapter"][2]
    floor = lambda j: u + h + ad + (j + 1) * bpe * pb
    s = []
    s.append(P("The mathematics of removing the cut:<br/>exactness, convex cost, "
               "and where the ceiling comes from", TITLE))
    s.append(P("Theory supplement to the FLEX-UF early-exit decoder. Notation "
               "follows <font face='Courier'>paper/theory.tex</font>; this "
               "document extends it to the regime that document's first "
               "assumption excludes.", AUTH))

    s.append(P("<b>Abstract.</b> The tiled multi-exit decoder is analysed in the "
               "main paper under two assumptions: the compute of a frame is the "
               "mean of per-tile costs, and the distortion is the mean of "
               "per-tile distortions. The first is exact for a tiled decoder and "
               "false for a decoder that does not tile. We replace tiling with a "
               "depth field and a max-plus dilation, and prove four things. "
               "(i) The decode is <i>exact</i>: every position receives what a "
               "full-frame decode to its own depth would have produced, with no "
               "value read from outside the frame (Theorem 1). (ii) The "
               "distortion becomes exactly separable in the depth field "
               "(Theorem 2), which is what makes the oracle table computable "
               "from K full-frame decodes and is false under tiling. (iii) The "
               "compute stops being additive and becomes a convex function of "
               "the depth field, whose linearisation is precisely the additive "
               "proxy the allocator uses; the gap is the band, and it bounds how "
               "far the practical allocator can be from optimal (Theorem 4). "
               "(iv) The reachable saving is capped by a floor that the split "
               "depth inflates, and removing the split moves the ceiling from "
               f"{100*(1-floor(2)):.1f}% to {100*(1-floor(0)):.1f}% "
               "(Proposition 8). A fifth result explains an empirical finding: "
               "when the exit-error curve is rank-1, the oracle allocation is a "
               "threshold rule on a single per-cell statistic, so a router needs "
               "the right <i>ranking</i> and nothing else (Theorem 6).", ABST))

    # ---------------------------------------------------------------- 1
    s.append(P("1&nbsp;&nbsp;Setup", H1))
    s.append(P("Let &Omega; be the feature grid of one frame and let the trunk be "
               "a composition of <i>L</i> blocks "
               "<i>G</i><sub>1</sub>,&hellip;,<i>G<sub>L</sub></i>, each a "
               "3&times;3 operation, so that block <i>b</i> at position <i>x</i> "
               "reads only positions within Chebyshev distance 1 of <i>x</i> in "
               "the state after block <i>b</i>&minus;1. Write "
               "<i>F<sub>b</sub></i> for the full-frame state after <i>b</i> "
               "blocks, computed everywhere. In the measured system "
               "<i>L</i>&nbsp;=&nbsp;12, exits are taken every "
               "<i>&beta;</i>&nbsp;=&nbsp;2 blocks, and "
               "<i>K</i>&nbsp;=&nbsp;6."))
    s.append(P("A <b>depth field</b> is a map <i>d</i>:&nbsp;&Omega;&nbsp;&rarr;"
               "&nbsp;{0,&hellip;,<i>L</i>} giving the number of blocks each "
               "position is allocated. Tiling is the special case in which "
               "<i>d</i> is constant on each tile. Nothing below assumes that."))
    s.append(P("2&nbsp;&nbsp;The dilation", H1))
    s.append(P("Define"))
    s.append(display(r"D(x)\;=\;\max_{y\in\Omega}\;\left[\,d(y)\;-\;"
                     r"\mathrm{dist}(x,y)\,\right],"))
    s.append(P("with dist the Chebyshev distance on the grid. In max-plus "
               "algebra this is the grey-scale morphological dilation of "
               "<i>d</i> by a cone of unit slope; the identification is standard "
               "and gives the next lemma for free, but the proof is two lines so "
               "it is given."))
    s.append(theorem("Lemma", 1, "least 1-Lipschitz majorant",
                     "<i>D</i> is the pointwise smallest function that dominates "
                     "<i>d</i> and is 1-Lipschitz. That is: (i) "
                     "<i>D</i>&nbsp;&ge;&nbsp;<i>d</i>; (ii) "
                     "|<i>D</i>(<i>x</i>)&minus;<i>D</i>(<i>x</i>&prime;)|&nbsp;"
                     "&le;&nbsp;dist(<i>x</i>,<i>x</i>&prime;); and (iii) if "
                     "<i>f</i>&nbsp;&ge;&nbsp;<i>d</i> is 1-Lipschitz then "
                     "<i>f</i>&nbsp;&ge;&nbsp;<i>D</i>."))
    s.append(proof(
        "(i) take <i>y</i>&nbsp;=&nbsp;<i>x</i>. (ii) for any <i>y</i>, "
        "dist(<i>x</i>,<i>y</i>)&nbsp;&le;&nbsp;dist(<i>x</i>&prime;,<i>y</i>)"
        "&nbsp;+&nbsp;dist(<i>x</i>,<i>x</i>&prime;), so "
        "<i>d</i>(<i>y</i>)&minus;dist(<i>x</i>,<i>y</i>)&nbsp;&ge;&nbsp;"
        "<i>d</i>(<i>y</i>)&minus;dist(<i>x</i>&prime;,<i>y</i>)&minus;"
        "dist(<i>x</i>,<i>x</i>&prime;); take the max over <i>y</i> and repeat "
        "with the roles exchanged. (iii) 1-Lipschitz gives "
        "<i>f</i>(<i>x</i>)&nbsp;&ge;&nbsp;<i>f</i>(<i>y</i>)&minus;"
        "dist(<i>x</i>,<i>y</i>)&nbsp;&ge;&nbsp;<i>d</i>(<i>y</i>)&minus;"
        "dist(<i>x</i>,<i>y</i>) for every <i>y</i>."))
    s.append(P("Property (ii) is the one that does the work below: the dilated "
               "field cannot fall by more than one block per step, which is "
               "exactly the rate at which a 3&times;3 receptive field grows."))

    # ---------------------------------------------------------------- 3
    s.append(P("3&nbsp;&nbsp;Exactness", H1))
    s.append(P("Let the decoder compute block <i>b</i> on the set "
               "<i>S<sub>b</sub></i>&nbsp;=&nbsp;{<i>x</i>&nbsp;:&nbsp;"
               "<i>D</i>(<i>x</i>)&nbsp;&ge;&nbsp;<i>b</i>} and nowhere else, "
               "leaving positions outside <i>S<sub>b</sub></i> holding their "
               "value from block <i>b</i>&minus;1. Write "
               "<i>&Phi;<sub>b</sub></i>(<i>x</i>) for the value this procedure "
               "holds at <i>x</i> after block <i>b</i>."))
    s.append(theorem("Theorem", 1, "exact per-position decoding",
                     "For every <i>b</i> and every <i>x</i> with "
                     "<i>D</i>(<i>x</i>)&nbsp;&ge;&nbsp;<i>b</i>, "
                     "<i>&Phi;<sub>b</sub></i>(<i>x</i>)&nbsp;=&nbsp;"
                     "<i>F<sub>b</sub></i>(<i>x</i>). In particular each "
                     "position's read-out at its own depth, "
                     "<i>&Phi;<sub>d(x)</sub></i>(<i>x</i>), equals "
                     "<i>F<sub>d(x)</sub></i>(<i>x</i>): what a full-frame decode "
                     "to depth <i>d</i>(<i>x</i>) would have produced there."))
    s.append(proof(
        "Induction on <i>b</i>. For <i>b</i>&nbsp;=&nbsp;0 the state is the "
        "upsampled latent, computed everywhere, so the claim holds. Assume it "
        "for <i>b</i>&minus;1 and take <i>x</i> with "
        "<i>D</i>(<i>x</i>)&nbsp;&ge;&nbsp;<i>b</i>. Block <i>b</i> at <i>x</i> "
        "reads only positions <i>y</i> with dist(<i>x</i>,<i>y</i>)&nbsp;&le;"
        "&nbsp;1. By Lemma 1(ii), <i>D</i>(<i>y</i>)&nbsp;&ge;&nbsp;"
        "<i>D</i>(<i>x</i>)&minus;1&nbsp;&ge;&nbsp;<i>b</i>&minus;1, so every "
        "such <i>y</i> lies in <i>S</i><sub><i>b</i>&minus;1</sub> and by the "
        "induction hypothesis holds <i>F</i><sub><i>b</i>&minus;1</sub>(<i>y</i>). "
        "The block therefore receives exactly the inputs the full-frame "
        "computation would give it, and being a deterministic function of them, "
        "returns <i>F<sub>b</sub></i>(<i>x</i>). The final claim follows from "
        "<i>d</i>&nbsp;&le;&nbsp;<i>D</i> (Lemma 1(i))."))
    s.append(theorem("Corollary", 2, "no invented values",
                     "The procedure never reads a value outside &Omega;, and in "
                     "particular never reads a padded one. A tiled decoder reads "
                     "one padded value per border position per block."))
    s.append(P("Corollary 2 is the formal content of the phrase &ldquo;there is "
               "no seam&rdquo;. It is not that the seam is small; it is that the "
               "quantity a seam measures does not exist in this procedure, at any "
               "granularity of <i>d</i>."))
    return s


def story2(s):
    c = C()
    u, h, tr = c["SHARE_UPSAMPLE"], c["SHARE_HEAD"], c["SHARE_TRUNK"]
    pb, bpe, K, N = c["per_block"], c["blocks_per_exit"], c["K"], c["N_TRUNK_BLOCKS"]
    ad = c["adapter"][2]
    floor = lambda j: u + h + ad + (j + 1) * bpe * pb

    # ---------------------------------------------------------------- 4
    s.append(P("4&nbsp;&nbsp;Distortion becomes separable", H1))
    s.append(P("Under tiling, the reconstruction on tile <i>i</i> depends on the "
               "depths of neighbouring tiles, because the padding a border reads "
               "is a function of where the cut was made. That is why the main "
               "paper's Assumption 2 is stated as an assumption. Under Theorem 1 "
               "it becomes a consequence."))
    s.append(theorem("Theorem", 2, "separability",
                     "Let the read-out at <i>x</i> be "
                     "<i>H</i>(<i>A</i><sub><i>d(x)</i></sub>"
                     "<i>&Phi;</i><sub><i>d(x)</i></sub>(<i>x</i>)) with "
                     "<i>A<sub>k</sub></i> the exit-<i>k</i> adapter and "
                     "<i>H</i> the head applied pointwise. Then the frame "
                     "distortion is separable in the depth field:"))
    s.append(display(r"\mathcal{D}(d)\;=\;\sum_{x\in\Omega}\,m\!\left(x,\,"
                     r"d(x)\right),\qquad m(x,k)\;=\;\left\|\,H(A_k F_k(x))"
                     r"\;-\;t(x)\,\right\|^2 ,"))
    s.append(P("where <i>t</i> is the target. In particular <i>m</i>(&middot;,"
               "<i>k</i>) is obtained from <i>K</i> full-frame decodes, one per "
               "exit, and does not depend on the allocation at all."))
    s.append(proof(
        "immediate from Theorem 1: <i>&Phi;</i><sub><i>d(x)</i></sub>(<i>x</i>) "
        "= <i>F</i><sub><i>d(x)</i></sub>(<i>x</i>), a quantity determined by "
        "the latent and by <i>d</i>(<i>x</i>) alone."))
    s.append(P("<b>The head's receptive field.</b> The deployed decoder does not "
               "apply the head pointwise: it stitches the adapted features and "
               "runs one head over the frame, and that head has its own "
               "3&times;3. Separability is therefore exact for the trunk output "
               "and holds for the pixel output up to a coupling supported on the "
               "one-cell neighbourhood of a depth boundary. That residual is not "
               "assumed away; it is measured. On all 53 test frames at "
               "64&nbsp;px cells it is below 0.0002&nbsp;dB, and its sign is "
               "favourable &mdash; the deployed decoder is very slightly better "
               "than the separable idealisation, not worse."))
    s.append(P("Theorem 2 is what makes the rest of this project computable. "
               "Every oracle number quoted anywhere comes from tabulating "
               "<i>m</i>(<i>x</i>,<i>k</i>) once and minimising over "
               "allocations; under tiling that table does not exist, because "
               "<i>m</i> would depend on the neighbours' depths as well."))

    # ---------------------------------------------------------------- 5
    s.append(P("5&nbsp;&nbsp;Cost stops being additive and becomes convex", H1))
    s.append(P("The number of block-evaluations the procedure performs is"))
    s.append(display(r"\mathcal{C}_{\mathrm{true}}(d)\;=\;\sum_{b=1}^{L}"
                     r"\bigl|S_b\bigr|\;=\;\sum_{x\in\Omega} D(x),"
                     .replace(r"\bigl|", "|").replace(r"\bigr|", "|")))
    s.append(P("against the additive proxy the allocator actually optimises,"))
    s.append(display(r"\mathcal{C}_{\mathrm{sep}}(d)\;=\;\sum_{x\in\Omega} d(x)"
                     r"\;\leq\;\mathcal{C}_{\mathrm{true}}(d),"))
    s.append(P("with the inequality from Lemma 1(i). The difference is what this "
               "project calls the <b>band</b>. The next two results say what kind "
               "of object each side is."))
    s.append(theorem("Proposition", 3, "level sets of the dilation",
                     "For every integer <i>b</i>,"))
    s.append(display(r"S_b\;=\;\{x: D(x)\geq b\}\;=\;\bigcup_{c\,\geq\,b}\;"
                     r"\left(\,L_c \oplus B_{\,c-b}\,\right),\qquad "
                     r"L_c=\{y: d(y)\geq c\},"))
    s.append(P("where &oplus; is dilation by the ball <i>B<sub>r</sub></i> of "
               "radius <i>r</i>."))
    s.append(proof(
        "<i>D</i>(<i>x</i>)&nbsp;&ge;&nbsp;<i>b</i> holds iff there is a "
        "<i>y</i> with <i>d</i>(<i>y</i>)&nbsp;&ge;&nbsp;<i>b</i>&nbsp;+&nbsp;"
        "dist(<i>x</i>,<i>y</i>). Put <i>c</i>&nbsp;=&nbsp;<i>d</i>(<i>y</i>); "
        "then <i>c</i>&nbsp;&ge;&nbsp;<i>b</i> and dist(<i>x</i>,<i>y</i>)&nbsp;"
        "&le;&nbsp;<i>c</i>&minus;<i>b</i>, i.e. <i>x</i>&nbsp;&isin;&nbsp;"
        "<i>L<sub>c</sub></i>&nbsp;&oplus;&nbsp;<i>B</i><sub><i>c</i>&minus;"
        "<i>b</i></sub>. Conversely such an <i>x</i> admits a witness <i>y</i> "
        "with <i>d</i>(<i>y</i>)&minus;dist(<i>x</i>,<i>y</i>)&nbsp;&ge;&nbsp;"
        "<i>c</i>&minus;(<i>c</i>&minus;<i>b</i>)&nbsp;=&nbsp;<i>b</i>."))
    s.append(P("Proposition 3 is the formal version of &ldquo;the band's width "
               "is the depth <i>difference</i> between neighbours&rdquo;: a "
               "region allocated depth <i>c</i> forces block <i>b</i> to run only "
               "within <i>c</i>&minus;<i>b</i> of it, so a map whose depth varies "
               "slowly pays almost nothing. Together with a perimeter bound "
               "|<i>A</i>&nbsp;&oplus;&nbsp;<i>B<sub>r</sub></i>|&nbsp;&le;&nbsp;"
               "|<i>A</i>|&nbsp;+&nbsp;<i>r</i>&thinsp;<i>P</i>(<i>A</i>)&nbsp;+"
               "&nbsp;<i>O</i>(<i>r</i>&sup2;) it gives the band a bound in terms "
               "of the perimeters of the depth level sets, which is why the "
               "measured band grows roughly with the inverse of the cell side: "
               f"{J('granularity_ctc53_fixed.json')['rows'][0]['cells'][0]['band_cost_pct']:.2f} "
               "points of saving at 256&nbsp;px against "
               f"{J('granularity_ctc53_fixed.json')['rows'][0]['cells'][3]['band_cost_pct']:.2f} "
               "at 32&nbsp;px, at the lowest rate."))
    s.append(theorem("Theorem", 4, "convexity, and an a-posteriori optimality bound",
                     "Regard <i>d</i> as a vector in "
                     "&#8477;<sup>|&Omega;|</sup>. Then "
                     "<i>C</i><sub>true</sub> is convex, "
                     "<i>C</i><sub>sep</sub> is linear, and "
                     "<i>C</i><sub>sep</sub> is the largest linear minorant of "
                     "<i>C</i><sub>true</sub> that is exact on constant fields. "
                     "Moreover, if <i>d&#770;</i> minimises the separable "
                     "Lagrangian <i>J</i><sub>sep</sub>(<i>d</i>)&nbsp;=&nbsp;"
                     "<i>D</i>(<i>d</i>)&nbsp;+&nbsp;&lambda;"
                     "<i>C</i><sub>sep</sub>(<i>d</i>), then"))
    s.append(display(r"J_{\mathrm{true}}(\hat d)\;-\;\min_{d}\,"
                     r"J_{\mathrm{true}}(d)\;\;\leq\;\;\lambda\,\cdot\,"
                     r"\mathrm{band}(\hat d)."))
    s.append(proof(
        "<i>D</i>(&middot;) is a pointwise maximum of affine functions of "
        "<i>d</i>, hence convex, and a sum of convex functions is convex; "
        "<i>C</i><sub>sep</sub> is a sum of coordinates, hence linear; on a "
        "constant field <i>D</i>&nbsp;&equiv;&nbsp;<i>d</i> so the two agree. "
        "For the bound, write "
        "<i>J</i><sub>true</sub>&nbsp;=&nbsp;<i>J</i><sub>sep</sub>&nbsp;+&nbsp;"
        "&lambda;&nbsp;band. Then "
        "min<sub><i>d</i></sub>&nbsp;<i>J</i><sub>true</sub>&nbsp;&ge;&nbsp;"
        "min<sub><i>d</i></sub>&nbsp;<i>J</i><sub>sep</sub>&nbsp;=&nbsp;"
        "<i>J</i><sub>sep</sub>(<i>d&#770;</i>), and "
        "<i>J</i><sub>true</sub>(<i>d&#770;</i>)&nbsp;=&nbsp;"
        "<i>J</i><sub>sep</sub>(<i>d&#770;</i>)&nbsp;+&nbsp;&lambda;&nbsp;"
        "band(<i>d&#770;</i>). Subtract."))
    s.append(P("The bound is computable after the fact, because the band of the "
               "chosen map is measured by dilating it. It also predicts a "
               "negative result this project obtained empirically: a "
               "smoothness-regularised allocation, which is an attempt to spend "
               "less band, cannot gain more than &lambda;&thinsp;band and was "
               "measured to gain less than nothing &mdash; it halved the band and "
               "lost slightly more in distortion."))
    return s


def story3(s):
    c = C()
    u, h = c["SHARE_UPSAMPLE"], c["SHARE_HEAD"]
    pb, bpe, K = c["per_block"], c["blocks_per_exit"], c["K"]
    ad = c["adapter"][2]
    floor = lambda j: u + h + ad + (j + 1) * bpe * pb

    # ---------------------------------------------------------------- 6
    s.append(P("6&nbsp;&nbsp;The allocation", H1))
    s.append(P("With Theorem 2 the distortion is separable and with "
               "<i>C</i><sub>sep</sub> the cost is too, so the Lagrangian "
               "decouples and the per-cell argmin is exactly optimal for it. "
               "This is Everett's theorem specialised to our setting, and it is "
               "the same argument the main paper's Theorem 2 makes for tiles; "
               "what is new is that here it is exact rather than an assumption."))
    s.append(theorem("Proposition", 5, "decoupling",
                     "For every &lambda;&nbsp;&ge;&nbsp;0,"))
    s.append(display(r"\min_{d}\;\left[\,\mathcal{D}(d)+\lambda\,"
                     r"\mathcal{C}_{\mathrm{sep}}(d)\,\right]\;=\;"
                     r"\sum_{x\in\Omega}\;\min_{k\in\mathcal{K}}\;"
                     r"\left[\,m(x,k)+\lambda\,c_k\,\right],"))
    s.append(P("attained by <i>d</i>&#770;(<i>x</i>)&nbsp;&isin;&nbsp;"
               "argmin<sub><i>k</i></sub>&nbsp;[<i>m</i>(<i>x</i>,<i>k</i>)"
               "&nbsp;+&nbsp;&lambda;<i>c<sub>k</sub></i>]. Sweeping &lambda; "
               "traces the lower convex envelope of the achievable "
               "(cost,&nbsp;distortion) set, and only that envelope; points "
               "strictly inside a facet are reached by mixing two &lambda; "
               "values, not by any single one."))
    s.append(P("<b>A caution the implementation had to learn.</b> The feasible "
               "set is <i>K</i>&nbsp;=&nbsp;{<i>j</i>,&hellip;,<i>K</i>&minus;1} "
               "and the deployed <font face='Courier'>forward()</font> clamps a "
               "map into it. Taking the argmin over all <i>K</i> exits and "
               "clamping afterwards is <i>not</i> the same optimisation: it "
               "prices the shallow options at costs "
               "<i>c</i><sub>0</sub>,&nbsp;<i>c</i><sub>1</sub> that no decoder "
               "charges, and biases every allocation shallow. The two coincide "
               "only when the cost vector is itself clamped, which the shipped "
               "<font face='Courier'>exit_costs</font> does and a hand-written "
               "block-count does not."))
    s.append(P("6.1&nbsp;&nbsp;Why the ranking is all a router needs", H2))
    s.append(P("The empirical finding that motivates this subsection: on the "
               "test set the log of the exit-error curve is rank-1 to 91.7% of "
               "its variance, and a parameter-free surrogate that only ranks "
               "cells attains 83&ndash;100% of the oracle's saving. The following "
               "explains why that is not a coincidence."))
    s.append(theorem("Theorem", 6, "threshold structure under a rank-1 ladder",
                     "Suppose <i>m</i>(<i>x</i>,<i>k</i>)&nbsp;=&nbsp;"
                     "<i>s</i>(<i>x</i>)&thinsp;&phi;<sub><i>k</i></sub> with "
                     "&phi; strictly decreasing and <i>c</i> strictly increasing "
                     "in <i>k</i>. Then <i>k</i>*(<i>x</i>)&nbsp;=&nbsp;"
                     "argmin<sub><i>k</i></sub>[<i>s</i>(<i>x</i>)&phi;<sub>"
                     "<i>k</i></sub>&nbsp;+&nbsp;&lambda;<i>c<sub>k</sub></i>] is "
                     "non-decreasing in <i>s</i>(<i>x</i>). Consequently the "
                     "oracle allocation is a threshold rule: there are cut points "
                     "<i>&tau;</i><sub><i>j</i></sub>&nbsp;&le;&nbsp;&hellip;&nbsp;"
                     "&le;&nbsp;<i>&tau;</i><sub><i>K</i>&minus;2</sub> such that "
                     "<i>k</i>*(<i>x</i>) is determined by which interval "
                     "<i>s</i>(<i>x</i>) falls in."))
    s.append(proof(
        "for <i>k</i>&prime;&nbsp;&gt;&nbsp;<i>k</i> the difference of the two "
        "objectives is <i>&Delta;</i>(<i>s</i>)&nbsp;=&nbsp;<i>s</i>(&phi;<sub>"
        "<i>k</i>&prime;</sub>&minus;&phi;<sub><i>k</i></sub>)&nbsp;+&nbsp;"
        "&lambda;(<i>c</i><sub><i>k</i>&prime;</sub>&minus;<i>c<sub>k</sub></i>). "
        "Since &phi;<sub><i>k</i>&prime;</sub>&nbsp;&lt;&nbsp;&phi;<sub><i>k</i>"
        "</sub>, <i>&Delta;</i> is strictly decreasing in <i>s</i>, so it "
        "changes sign at most once and only from + to &minus;. This is the "
        "single-crossing condition, and the monotone comparative statics "
        "conclusion follows."))
    s.append(P("The practical content: under a rank-1 ladder a router does not "
               "need a calibrated estimate of the error at all. It needs the "
               "correct <i>order</i> of cells by <i>s</i>, plus "
               "<i>K</i>&minus;<i>j</i>&minus;1 thresholds, which the "
               "&lambda;-sweep supplies. Any strictly monotone transform of "
               "<i>s</i> is as good as <i>s</i>. That is precisely the freedom a "
               "rate-rank surrogate exploits, and it is why replacing it with a "
               "regression that predicts <i>s</i> accurately buys so little: the "
               "measured gain of the best learned model over the rank-1 surrogate "
               "is 0.68 points against an oracle gap of 3.3."))
    s.append(theorem("Proposition", 7, "misallocation costs the Lagrangian gap, "
                     "not the error rate",
                     "If a router assigns <i>k&#770;</i>(<i>x</i>) instead of "
                     "<i>k</i>*(<i>x</i>), the excess objective is"))
    s.append(display(r"\mathcal{E}\;=\;\sum_{x\in\Omega}\Bigl(\,\left[m(x,\hat "
                     r"k)+\lambda c_{\hat k}\right]-\left[m(x,k^{*})+\lambda "
                     r"c_{k^{*}}\right]\Bigr)\;\;\geq\;\;0 ,"
                     .replace(r"\Bigl(", "(").replace(r"\Bigr)", ")")))
    s.append(P("which vanishes on every cell where the two exits are tied, "
               "however many such cells are mislabelled. Agreement with the "
               "oracle is therefore not the loss a router should be measured on, "
               "and a router with lower agreement can have lower excess. Every "
               "router number in the companion report is scored by the "
               "allocation it causes, not by agreement."))
    return s


def story4(s):
    c = C()
    u, h = c["SHARE_UPSAMPLE"], c["SHARE_HEAD"]
    pb, bpe, K = c["per_block"], c["blocks_per_exit"], c["K"]
    ad = c["adapter"][2]
    seam = 1.0095 - (u + c["SHARE_TRUNK"] + h)
    floor = lambda j, sr=0.0: u + h + ad + (j + 1) * bpe * pb + sr

    # ---------------------------------------------------------------- 7
    s.append(P("7&nbsp;&nbsp;The floor, the ceiling, and when the ceiling is "
               "out of reach", H1))
    s.append(P("Everything above concerns which allocation is best. This section "
               "is about what the best one can possibly be worth, which is fixed "
               "by the cheapest legal allocation."))
    s.append(theorem("Proposition", 8, "floor and ceiling",
                     "Let the feasible exits be {<i>j</i>,&hellip;,"
                     "<i>K</i>&minus;1}. The cost of any allocation is at least"))
    s.append(display(r"\Phi(j)\;=\;u\;+\;(j+1)\,\beta\,p\;+\;a\;+\;h\;+\;"
                     r"\sigma ,"))
    s.append(P("&mdash;&nbsp;the upsample <i>u</i>, the "
               "(<i>j</i>+1)&thinsp;&beta; trunk blocks every position must run, "
               "the adapter <i>a</i>, the head <i>h</i>, and the seam-repair "
               "share &sigma; &mdash;", ParagraphStyle(
                   "note", parent=BODY, alignment=1, fontSize=8.8,
                   textColor=INK2, spaceAfter=6)))
    s.append(P("with &beta; blocks per exit, <i>p</i> the per-block share and "
               "&sigma; the seam-repair share, and the saving is therefore at "
               "most 1&nbsp;&minus;&nbsp;&Phi;(<i>j</i>). The bound is attained "
               "exactly when every position takes exit <i>j</i>."))
    s.append(proof(
        "the upsample and the head run once per frame regardless of the map, "
        "every position runs at least (<i>j</i>+1)&beta; blocks because the "
        "feasible set starts at <i>j</i>, and every position that is not at the "
        "deepest exit wears an adapter. Summing and using "
        "<i>C</i><sub>true</sub>&nbsp;&ge;&nbsp;<i>C</i><sub>sep</sub> gives the "
        "bound; on a constant field the two costs agree, which gives "
        "attainment."))
    rows = [["configuration", "&Phi;", "ceiling"]]
    rows.append(["tiled, <i>j</i>=2 (with seam repair)",
                 f"{floor(2, seam):.4f}", f"{100*(1-floor(2, seam)):.2f}%"])
    rows.append(["per position, <i>j</i>=2", f"{floor(2):.4f}",
                 f"{100*(1-floor(2)):.2f}%"])
    rows.append(["per position, <i>j</i>=1", f"{floor(1):.4f}",
                 f"{100*(1-floor(1)):.2f}%"])
    rows.append(["per position, <i>j</i>=0", f"{floor(0):.4f}",
                 f"{100*(1-floor(0)):.2f}%"])
    s.append(table(rows, "floor",
                   "Proposition 8 instantiated with the measured constants "
                   f"<i>u</i>={u}, <i>p</i>={pb:.5f}, &beta;={bpe}, "
                   f"<i>a</i>={ad:.5f}, <i>h</i>={h}, &sigma;={seam:.4f}. The "
                   "split depth costs "
                   f"{100*(floor(2)-floor(0)):.1f} points of ceiling, and seam "
                   f"repair a further {100*seam:.2f}.",
                   widths=[3.1 * 72, 1.1 * 72, 1.3 * 72], align_right_from=1))
    s.append(theorem("Proposition", 9, "feasibility of the ceiling",
                     "At a distortion budget &beta;<sub>dB</sub> the ceiling is "
                     "attainable if and only if the uniform allocation "
                     "<i>d</i>&nbsp;&equiv;&nbsp;<i>j</i> already meets the "
                     "budget. Otherwise the optimum lies at an interior &lambda; "
                     "and the reachable saving is strictly below the ceiling."))
    s.append(proof(
        "the ceiling is attained only by <i>d</i>&nbsp;&equiv;&nbsp;<i>j</i> "
        "(Proposition 8), so it is feasible exactly when that allocation is. If "
        "it is not, the constraint binds, and by Proposition 5 the constrained "
        "optimum is the &lambda; at which the budget holds with equality."))
    s.append(P("On the measured system the uniform-<i>j</i> allocation costs "
               "0.107&nbsp;dB at the lowest rate and 0.248 at the highest, "
               "against a 0.1&nbsp;dB budget, so Proposition 9 says the ceiling "
               "is not reachable at that budget at any rate &mdash; which is why "
               "the realised saving sits at 70&ndash;90% of it and why a looser "
               "0.3&nbsp;dB budget saturates instead of improving."))

    # ---------------------------------------------------------------- 8
    s.append(P("8&nbsp;&nbsp;What the three changes are, in these terms", H1))
    s.append(P("The companion report measures three inference-time changes. In "
               "the language above they are exactly:"))
    s.append(P("<b>(i)</b> Replace the tiled decode by the dilated one. This "
               "converts Assumption 2 of the main paper into Theorem 2, removes "
               "&sigma; from &Phi; by Corollary 2, and replaces the additive cost "
               "by <i>C</i><sub>true</sub>, which exceeds it by the band."))
    s.append(P("<b>(ii)</b> Refine the depth field. Under tiling the cell size is "
               "bounded below by the seam it creates; under Corollary 2 there is "
               "no such bound, and by Proposition 3 the price of refinement is "
               "the perimeter term alone. The optimum over cell sizes is interior "
               "and was located empirically at 64&nbsp;px."))
    s.append(P("<b>(iii)</b> Take <i>j</i>&nbsp;=&nbsp;0. The split exists to "
               "bound the seam; by Corollary 2 the seam is identically zero, so "
               "the constraint has no purpose and Proposition 8 gives back "
               f"{100*(floor(2)-floor(0)):.1f} points of ceiling. Whether that "
               "translates into realised saving depends on whether the shallow "
               "rungs are good enough for Proposition 9 to bite less hard, which "
               "is an empirical question the companion report answers per rate."))
    s.append(P("9&nbsp;&nbsp;Relation to the main paper's theory", H1))
    s.append(P("The main paper proves that per-tile assignment achieves the "
               "Minkowski average of the per-tile achievable sets, that the "
               "adaptivity gain &Delta;(&lambda;) is non-negative and vanishes "
               "only when one exit is optimal for every tile, and that the "
               "frontier is convex in the units it is plotted in. All three "
               "survive verbatim with tiles replaced by positions, because they "
               "depend only on separability of the objective &mdash; which "
               "Theorem 2 supplies exactly where the tiled version assumed it. "
               "What does not survive is cost additivity, and Theorem 4 is the "
               "replacement: the true cost is convex, the additive form is its "
               "tight linear minorant, and the allocator that uses the minorant "
               "is suboptimal by at most &lambda; times a band it can measure."))

    s.append(P("References", H1))
    refs = [
        "Everett. Generalized Lagrange multiplier method for solving problems of "
        "optimum allocation of resources. <i>Operations Research</i>, 1963.",
        "Shoham and Gersho. Efficient bit allocation for an arbitrary set of "
        "quantizers. <i>IEEE Trans. ASSP</i>, 1988.",
        "Bjontegaard. Calculation of average PSNR differences between RD-curves. "
        "VCEG-M33, 2001.",
        "Maragos. Morphological signal processing and the slope transform. "
        "<i>Signal Processing</i>, 1994.",
        "Dorst and van den Boomgaard. Morphological signal processing and the "
        "slope transform. 1994.",
        "Serra. <i>Image Analysis and Mathematical Morphology</i>. 1982.",
        "Topkis. Minimizing a submodular function on a lattice. "
        "<i>Operations Research</i>, 1978.",
        "Milgrom and Shannon. Monotone comparative statics. "
        "<i>Econometrica</i>, 1994.",
        "Teerapittayanon, McDanel and Kung. BranchyNet: fast inference via early "
        "exiting from deep neural networks. ICPR 2016.",
        "Huang et&nbsp;al. Multi-scale dense networks for resource efficient "
        "image classification. ICLR 2018.",
        "Bengio, L&eacute;onard and Courville. Estimating or propagating "
        "gradients through stochastic neurons for conditional computation. 2013.",
        "Schuster et&nbsp;al. Confident Adaptive Language Modeling. NeurIPS 2022.",
        "Schuster et&nbsp;al. Consistent accelerated inference via confident "
        "adaptive transformers. EMNLP 2021.",
        "Zhang et&nbsp;al. Be Your Own Teacher: improve the performance of "
        "convolutional neural networks via self distillation. ICCV 2019.",
        "Liu et&nbsp;al. Deep adaptive inference networks for single image "
        "super-resolution. ECCV 2020.",
        "Adaptive patch exiting for scalable single image super-resolution. "
        "ECCV 2022.",
        "Mao et&nbsp;al. AdaRevD: adaptive patch exiting reversible decoder. "
        "CVPR 2024.",
        "Chu et&nbsp;al. Improving image restoration by revisiting global "
        "information aggregation. ECCV 2022.",
    ]
    for i, r in enumerate(refs, 1):
        s.append(P(f"[{i}]&nbsp;&nbsp;{r}",
                   ParagraphStyle("ref", parent=BODY, fontSize=8.6, leading=10.4,
                                  leftIndent=14, firstLineIndent=-14,
                                  spaceAfter=2.4)))
    return s


if __name__ == "__main__":
    st = story(); st = story2(st); st = story3(st); st = story4(st)
    out = Path(__file__).resolve().parent / "FLEX-UF-theory.pdf"
    build(st, out)
    print(f"  yazildi {out}")
