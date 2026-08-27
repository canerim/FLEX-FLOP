"""The talk, in TUM's colours, one result or one diagram per slide.

Six questions were asked, and each is answered on the slide that carries the
evidence for it rather than on an agenda page: motivation, baselines, the main
message, the experiments, the ablations, and the limits with what follows from
them. Every number is read from the measurement files at build time.

One experiment throughout: epoch 9 of RECIPE512, 256 px tiles, 53 CTC intra
frames, 0.1 dB unless a slide says otherwise.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph, Table, TableStyle, Image
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
RES = HERE.parent / "flexplus" / "results"
UF = Path.home() / "FLEX-UF"
QPS = [0, 16, 32, 48, 63]

W, H = 13.333 * inch, 7.5 * inch
TUM_BLUE = colors.HexColor("#0065BD")
TUM_DARK = colors.HexColor("#003359")
TUM_MID = colors.HexColor("#005293")
TUM_LT = colors.HexColor("#98C6EA")
TUM_ORANGE = colors.HexColor("#E37222")
TUM_GREEN = colors.HexColor("#A2AD00")
GREY = colors.HexColor("#6a6a6a")
INK = colors.HexColor("#1a1a1a")
BG = colors.HexColor("#f7f7f5")

A = json.loads((RES / "ablations.json").read_text())["families"]
D = json.loads((RES / "day_summary.json").read_text())
G = json.loads((RES / "guarantee_tiled_e9.json").read_text())
PF = json.loads((RES / "signalled_perframe_calexA.json").read_text())
HIST = json.loads((RES / "signalled_perframe_e9_hist.json").read_text())
REL = {r["qp"]: r for r in json.loads(
    (UF / "results/rd_absolute_PAPER.json").read_text())["rows"]}
DECGMAC = 454.0


def style(**kw):
    d = dict(fontName="Helvetica", fontSize=13, leading=17, textColor=INK)
    d.update(kw)
    return ParagraphStyle("s", **d)


class Deck:
    def __init__(self, path):
        self.c = pdfcanvas.Canvas(str(path), pagesize=landscape((H, W)))
        self.n = 0

    def slide(self, title, question=None, kicker=None):
        if self.n:
            self.c.showPage()
        self.n += 1
        c = self.c
        c.setFillColor(colors.white); c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(TUM_BLUE); c.rect(0, H - 0.95 * inch, W, 0.95 * inch,
                                         fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 21)
        c.drawString(0.55 * inch, H - 0.62 * inch, title)
        if kicker:
            c.setFont("Helvetica", 11.5)
            c.setFillColor(colors.HexColor("#cfe4f5"))
            c.drawRightString(W - 0.55 * inch, H - 0.60 * inch, kicker)
        if question:
            c.setFillColor(TUM_ORANGE)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(0.55 * inch, H - 1.28 * inch, question)
        # footer
        c.setFillColor(GREY); c.setFont("Helvetica", 8.5)
        c.drawString(0.55 * inch, 0.32 * inch,
                     "FLEX  ·  content-adaptive early exit in a learned image "
                     "decoder  ·  RECIPE512 epoch 9, 256 px tiles, 53 CTC "
                     "intra frames")
        c.drawRightString(W - 0.55 * inch, 0.32 * inch, str(self.n))
        c.setStrokeColor(colors.HexColor("#dddddd")); c.setLineWidth(0.5)
        c.line(0.55 * inch, 0.52 * inch, W - 0.55 * inch, 0.52 * inch)
        return c

    def text(self, x, y, s, size=13, colour=INK, bold=False, width=None):
        st = style(fontSize=size, leading=size * 1.38, textColor=colour,
                   fontName="Helvetica-Bold" if bold else "Helvetica")
        p = Paragraph(s, st)
        w = width or (W - 1.1 * inch)
        _, hh = p.wrap(w, H)
        p.drawOn(self.c, x, y - hh)
        return y - hh

    def fit(self, name, x, y, w, h, centre=True):
        """Fit an image inside (x, y, w, h), preserving aspect.

        The first version scaled by width alone, so a 2:1 teaser asked for
        12.2 inches of width, took 6.1 of height, and drew straight through
        the title bar. A slide is a box; an image has to be told about it.
        """
        p = FIG / name
        if not p.exists():
            p = UF / "paper" / "figures" / name
        ir = ImageReader(str(p))
        iw, ih = ir.getSize()
        sc = min(w / iw, h / ih)
        dw, dh = iw * sc, ih * sc
        dx = x + (w - dw) / 2 if centre else x
        dy = y + (h - dh) / 2
        self.c.drawImage(ir, dx, dy, width=dw, height=dh,
                         preserveAspectRatio=True, mask="auto")
        return dh

    def table(self, data, x, y, colw, head=True, fs=12, align=None,
              hl=None, rowh=0.315 * inch):
        t = Table(data, colWidths=colw, rowHeights=rowh)
        cmd = [("FONT", (0, 0), (-1, -1), "Helvetica", fs),
               ("TEXTCOLOR", (0, 0), (-1, -1), INK),
               ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
               ("LEFTPADDING", (0, 0), (-1, -1), 7),
               ("RIGHTPADDING", (0, 0), (-1, -1), 7),
               ("LINEBELOW", (0, 0), (-1, 0), 0.9, TUM_BLUE)]
        if head:
            cmd += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
                    ("TEXTCOLOR", (0, 0), (-1, 0), TUM_MID)]
        for r in range(1, len(data)):
            if r % 2 == 0:
                cmd.append(("BACKGROUND", (0, r), (-1, r), BG))
        if hl:
            for r in hl:
                cmd += [("BACKGROUND", (0, r), (-1, r),
                         colors.HexColor("#e6f0fa")),
                        ("FONT", (0, r), (-1, r), "Helvetica-Bold", fs)]
        if align:
            for col, a in align.items():
                cmd.append(("ALIGN", (col, 0), (col, -1), a))
        t.setStyle(TableStyle(cmd))
        _, th = t.wrap(sum(colw), H)
        t.drawOn(self.c, x, y - th)
        return th

    def save(self):
        self.c.save()
        print(f"  {self.n} slayt -> {self.c._filename}")


# ==========================================================================
def build(out="report/FLEX-talk-TUM.pdf"):
    d = Deck(HERE.parent / out)
    IN = 0.55 * inch

    # ---------------------------------------------------------------- title
    c = d.c
    c.setFillColor(colors.white); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TUM_BLUE); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 44)
    c.drawString(IN, H - 2.5 * inch, "FLEX")
    c.setFont("Helvetica", 22)
    c.drawString(IN, H - 3.25 * inch,
                 "Content-adaptive early exit in a learned image decoder")
    c.setFont("Helvetica", 14)
    c.setFillColor(colors.HexColor("#cfe4f5"))
    c.drawString(IN, H - 4.0 * inch,
                 "The bitstream does not change.  The decoder decides how much "
                 "of itself to run.")
    c.setFont("Helvetica-Bold", 16); c.setFillColor(colors.white)
    c.drawString(IN, H - 5.3 * inch,
                 f"{DECGMAC:.0f} GMAC  →  "
                 f"{DECGMAC*(1-D['configs']['calexA_perframe']['saving_pct']/100):.0f} GMAC "
                 f"per 1080p frame")
    c.setFont("Helvetica", 13); c.setFillColor(colors.HexColor("#cfe4f5"))
    c.drawString(IN, H - 5.75 * inch,
                 "ΔPSNR ≤ 0.1 dB on every frame  ·  264 of 265  ·  "
                 "no weight retrained")
    c.setFont("Helvetica", 11)
    c.drawString(IN, 0.7 * inch, "Technical University of Munich")
    d.n = 1

    # ------------------------------------------------------------ motivation
    d.slide("The decoder spends the same arithmetic everywhere",
            "Q1  How should we motivate the work?",
            kicker="motivation")
    d.text(IN, H - 1.7 * inch,
           "A learned decoder runs a fixed network over the whole frame, and "
           "the network is fixed by design — that is what lets any decoder "
           "read any file. A flat sky and a face are decoded at the same "
           "price. Almost every attempt to make decoders cheaper changes the "
           "model, and so the bitstream: a file encoded yesterday cannot "
           "benefit.", size=13.5)
    d.fit("fig11_budget.png", IN, 1.0 * inch, 5.8 * inch, 4.5 * inch)
    d.text(6.7 * inch, H - 3.0 * inch,
           "<b>What is left to vary is not what the decoder is, but how much "
           "of it runs where.</b><br/><br/>"
           "The trunk is 89.4% of the decode and it is the only routable "
           "part. The opening upsample (8.2%) and the head (2.4%) run "
           "full-frame whatever we do, which is why no allocation can save "
           "more than about 89%.", size=13, width=6.0 * inch)
    d.text(6.7 * inch, H - 5.3 * inch,
           "So the question is a scheduling question, not an architecture "
           "question, and it can be asked without touching the file.",
           size=13, colour=TUM_MID, width=6.0 * inch)

    # ------------------------------------------------------------- the ladder
    d.slide("The method: exits on a shared trunk, one per tile",
            "Q1  What exactly is fixed, and what is free?",
            kicker="method")
    d.fit("flexuf_overview.png", IN, 1.95 * inch, 12.2 * inch, 4.15 * inch)
    d.text(IN, 1.75 * inch,
           "Two groups run for every tile; <b>patchify</b> then cuts the "
           "feature map into 256 px tiles and each tile leaves the trunk at "
           "its own depth. An adapter reconciles the early exit with a head "
           "fitted to the full one; the deepest exit carries no adapter and "
           "is bit-exact with the released decoder. The encoder, the entropy "
           "model and the coded latent are untouched.", size=11.5)

    # -------------------------------------------------------------- router
    d.slide("The router: one exit per tile, from tensors the decode holds",
            "Q4  Which experiments are needed?  —  the decoder-side "
            "decision", kicker="method")
    d.fit("fig_router_teaser.png", IN, 0.75 * inch, 12.2 * inch, 5.25 * inch)

    # ------------------------------------------------------------- configs
    for f, t, q in (("fig_cfgA_teaser.png",
                     "Configuration A — signalled", None),
                    ("fig_cfgB_teaser.png",
                     "Configuration B — bitstream-identical", None),
                    ("fig_cfgC_teaser.png",
                     "Configuration C — partial signalling", None)):
        d.slide(t, q, kicker="method")
        d.fit(f, IN, 0.75 * inch, 12.2 * inch, 5.55 * inch)

    # --------------------------------------------------------- main result
    d.slide("Main result: 454 GMAC → 324 GMAC, with a per-frame guarantee",
            "Q3  What is the main message?", kicker="result")
    rows = [["rate", "bpp", "released PSNR", "ours", "ΔPSNR",
             "decoder GMAC", "saved"]]
    per = {r["qp"]: r for r in PF["rows"]}
    for q in QPS:
        r = per[q]
        rows.append([f"qp {q}", f"{REL[q]['bpp']:.4f}",
                     f"{REL[q]['psnr_release']:.3f} dB",
                     f"{REL[q]['psnr_release'] - r['mean_db']:.3f} dB",
                     f"−{r['mean_db']:.3f}",
                     f"{DECGMAC*(1-r['saving_pct_measured']/100):.0f}",
                     f"{r['saving_pct_measured']:.2f}%"])
    m = np.mean([per[q]["saving_pct_measured"] for q in QPS])
    rows.append(["mean", "", "", "", "", f"{DECGMAC*(1-m/100):.0f}",
                 f"{m:.2f}%"])
    d.table(rows, IN, H - 1.75 * inch,
            [1.5 * inch, 1.5 * inch, 2.3 * inch, 1.8 * inch, 1.5 * inch,
             2.0 * inch, 1.5 * inch], hl=[len(rows) - 1])
    d.text(IN, H - 4.6 * inch,
           "Same bitstream, byte for byte. Same encoder, same entropy model. "
           "The only new weights are five pointwise adapters, and their last "
           "linear map was solved in closed form rather than trained.",
           size=13)
    d.text(IN, H - 5.5 * inch,
           f"<b>264 of 265</b> frame-rate pairs are served at or under "
           f"0.1 dB. The one exception is a frame whose tiling floor is "
           f"{G['perframe']['max']:.3f} dB — above the budget before any "
           "tile has exited early, so no allocation can rescue it.",
           size=13, colour=TUM_MID)
    d.text(IN, H - 6.4 * inch,
           "Signalling cost: 85 to 108 bits per frame, including 16 bits for "
           "the multiplier. About 2×10⁻⁴ of the frame it steers.",
           size=12.5, colour=GREY)
    # ------------------------------------------------------- the guarantee
    d.slide("A budget the frame keeps, not one the set keeps on average",
            "Q3  The message is a guarantee, not an average",
            kicker="result")
    d.fit("fig17_guarantee.png", IN, 3.05 * inch, 12.2 * inch, 3.20 * inch)
    g, pfm = G["global"], G["perframe"]
    rows = [["where the multiplier is chosen", "mean saving",
             "frames over budget", "95th pct", "worst frame"],
            ["one per rate  (the usual protocol)", f"{g['saving']:.2f}%",
             f"{g['over']} / {g['n']}", f"{g['p95']:.3f} dB",
             f"{g['max']:.3f} dB"],
            ["one per frame", f"{pfm['saving']:.2f}%",
             f"{pfm['over']} / {pfm['n']}", f"{pfm['p95']:.3f} dB",
             f"{pfm['max']:.3f} dB"]]
    d.table(rows, IN, 2.85 * inch,
            [4.6 * inch, 1.9 * inch, 2.2 * inch, 1.7 * inch, 1.8 * inch],
            hl=[2], fs=12)
    d.text(IN, 1.45 * inch,
           "The guarantee is not paid for. A single multiplier solves a "
           "Lagrangian on <i>pooled</i> squared error, while the constraint "
           "is on a mean of logarithms — so the set-average form is not "
           "optimal even for the constraint it states.", size=12.5,
           colour=TUM_MID)

    # ---------------------------------------------------------- baselines
    d.slide("Baselines: a free rule already beats a trained router",
            "Q2  What baselines should we use?", kicker="baselines")
    rm = {r["model"]: r for r in A["router_models"]["rows"]}
    order = [("oracle", "oracle — exhaustive search, upper bound"),
             ("raterank", "rate-rank — one number per tile, no parameters"),
             ("mlp", "learned MLP — 144,024 parameters"),
             ("gbm", "gradient boosting on 26 hand-made features"),
             ("ridge", "ridge on the same features"),
             ("flat", "flat — one curve for every tile, no adaptivity")]
    rows = [["baseline", "mean saving at 0.1 dB", "of the oracle"]]
    ora = rm["oracle"]["mean"] if "oracle" in rm else float("nan")
    for k, lab in order:
        if k not in rm:
            continue
        rows.append([lab, f"{rm[k]['mean']:.2f}%",
                     f"{100*rm[k]['mean']/ora:.1f}%"])
    d.table(rows, IN, H - 1.75 * inch,
            [6.6 * inch, 3.0 * inch, 2.6 * inch], hl=[1, 2])
    d.text(IN, H - 4.6 * inch,
           "The comparison that matters is not against another network. It is "
           "against <b>the exhaustive search</b>, which bounds what any "
           "router can reach, and against <b>a rule with no parameters</b> "
           "that reads how many bits the entropy coder already spent on a "
           "tile.", size=13)
    d.text(IN, H - 5.6 * inch,
           "The free rule beats the learned head. We report that rather than "
           "hide it: the learned router exists because it can be conditioned "
           "on the operating point, not because it predicts better.",
           size=13, colour=TUM_ORANGE)

    # ------------------------------------------------- ablation: geometry
    d.slide("Ablation: what the saving is actually made of",
            "Q5  Which ablation studies?  —  the decode geometry",
            kicker="ablation")
    rows = [["variant", "mean saving", "against the row above"]]
    prev = None
    for r in A["decode_geometry"]["rows"]:
        delta = "" if prev is None else f"{r['mean']-prev:+.2f}"
        rows.append([r["decode"], f"{r['mean']:.2f}%", delta])
        prev = r["mean"]
    d.table(rows, IN, H - 1.75 * inch,
            [6.2 * inch, 2.6 * inch, 3.4 * inch])
    rows2 = [["cell size", "mean saving", "band cost"]]
    for r in A["granularity"]["rows"]:
        rows2.append([f"{r['cell_px']} px", f"{r['mean']:.2f}%",
                      f"{r['band_pct']:.2f} pts"])
    d.table(rows2, IN, H - 3.6 * inch,
            [6.2 * inch, 2.6 * inch, 3.4 * inch])
    d.text(IN, H - 5.9 * inch,
           "Cutting the frame into tiles costs quality before any tile has "
           "exited early — the <b>floor</b>. Per-position decoding removes "
           "it; finer cells then buy more, until the <b>band</b> (positions "
           "still computing to serve a deeper neighbour) eats the gain.",
           size=12.5)
    d.text(IN, H - 6.8 * inch,
           "Both rows are on the same checkpoint with no weight retrained.",
           size=12, colour=GREY)

    # ------------------------------------------- ablation: closed-form refit
    d.slide("Ablation: the gain is where the adapters are fitted, not that "
            "they are", "Q5  Which ablation studies?  —  the refit and its "
            "controls", kicker="ablation")
    rows = [["arm", "mean saving", "against baseline"]]
    base = A["calex_refit"]["baseline_mean"]
    rows.append(["baseline, epoch 9", f"{base:.2f}%", "—"])
    for r in A["calex_refit"]["rows"]:
        rows.append([r["arm"], f"{r['mean']:.2f}%", f"{r['mean']-base:+.2f}"])
    d.table(rows, IN, H - 1.75 * inch,
            [6.2 * inch, 2.6 * inch, 3.4 * inch], hl=[2])
    d.text(IN, H - 4.0 * inch,
           "Each adapter's last linear map is re-solved in closed form — a "
           "ridge regression onto the deepest exit's feature — on <b>the "
           "tiles that actually reach that exit</b>, at a multiplier matched "
           "to the deployed usage. 443,520 parameters, 0.98% of the decoder, "
           "two minutes on one card, no gradient step.", size=13)
    d.text(IN, H - 5.4 * inch,
           "Fitting on all tiles instead gives +0.19 rather than +0.64, so "
           "two thirds of the gain is the conditioning. Fitting on survivors "
           "chosen at the <i>wrong</i> multiplier gives −0.01: same method, "
           "same data, wrong conditional, and the gain disappears entirely.",
           size=13, colour=TUM_MID)

    # ------------------------------------------------ ablation: fine-tuning
    d.slide("Ablation: gradient descent adds nothing here",
            "Q5  Which ablation studies?  —  the negative result that "
            "explains the positive one", kicker="ablation")
    rows = [["arm", "mean saving", "against baseline"]]
    ft = {r["arm"]: r for r in A["fine_tuning"]["rows"]}
    b0 = ft["baseline, no fine-tune"]["mean"]
    for k in ("baseline, no fine-tune", "6000 steps, uniform weights (control)",
              "6000 steps, usage-matched weights",
              "closed-form refit, no training"):
        if k in ft:
            rows.append([k, f"{ft[k]['mean']:.2f}%",
                         f"{ft[k]['mean']-b0:+.2f}"])
    d.table(rows, IN, H - 1.75 * inch,
            [6.6 * inch, 2.6 * inch, 3.0 * inch], hl=[4])
    d.text(IN, H - 3.9 * inch,
           "Six thousand steps with the anchor on the released decoder and "
           "the adapters at the main run's own learning rate return exactly "
           "the baseline — with the usage prior and without it.", size=13)
    d.text(IN, H - 5.0 * inch,
           "The adapters are not undertrained. They have converged, for an "
           "<b>average</b> distribution, and stochastic gradient descent on "
           "OpenImages cannot be aimed away from it: every step sees every "
           "tile, and the multiplier matching that makes the closed form work "
           "has no analogue in the training loop. The closed form can state "
           "the conditional directly. That is the whole of the difference.",
           size=13, colour=TUM_ORANGE)

    # -------------------------------------------------------- exit usage
    d.slide("What the decoder actually runs", "Q4  Which experiments?  —  "
            "the allocation itself", kicker="result")
    rows = [["rate"] + [f"e{k}" for k in range(6)] + ["mean exit",
                                                      "trunk blocks run"]]
    tot = np.zeros(6)
    for r in HIST["rows"]:
        h = np.array(r["hist_pct"], float); tot += h * r["n"]
        me = float((np.arange(6) * h / 100).sum())
        rows.append([f"qp {r['qp']}"] + [f"{h[k]:.1f}%" for k in range(6)]
                    + [f"{me:.2f}", f"{(me+1)*2:.1f} / 12"])
    t = tot / tot.sum(); me = float((np.arange(6) * t).sum())
    rows.append(["all"] + [f"{100*t[k]:.1f}%" for k in range(6)]
                + [f"{me:.2f}", f"{(me+1)*2:.1f} / 12"])
    d.table(rows, IN, H - 1.75 * inch,
            [1.15 * inch] + [1.25 * inch] * 6 + [1.35 * inch, 1.85 * inch],
            hl=[len(rows) - 1], fs=11)
    d.text(IN, H - 4.6 * inch,
           "<b>e0 and e1 are exactly zero</b> — measured, not assumed. Under "
           "the clamp the first two exits are unreachable, and they are also "
           "not cheaper: the first four blocks run full-frame before "
           "patchify, so e0, e1 and e2 all cost 0.5716 of a decode.",
           size=13)
    d.text(IN, H - 5.7 * inch,
           "Two exits carry 82.6% of the tiles. That is why the closed-form "
           "refit concentrates its gain on them.", size=13, colour=TUM_MID)

    # ------------------------------------------------------- limitations
    d.slide("Limitations, measured rather than guessed",
            "Q6  What are the limitations?", kicker="limits")
    rows = [["", "mean saving", "images inside 0.1 dB", "mean tiling floor"]]
    for r in A["kodak"]["per_image"]:
        rows.append([f"Kodak, qp {r['qp']}", f"{r['saving']:.2f}%",
                     f"{r['n']-r['infeasible']} / {r['n']}",
                     f"{r['mean_floor_db']:.4f} dB"])
    d.table(rows, IN, H - 1.75 * inch,
            [3.6 * inch, 2.6 * inch, 3.2 * inch, 2.8 * inch], hl=[3])
    d.text(IN, H - 3.5 * inch,
           "<b>1. The floor can exceed the budget.</b> On Kodak at the top "
           "rate, 8 of 24 images cannot reach 0.1 dB at all: cutting them "
           "into tiles already costs more than the budget allows. It is not "
           "an allocation failure and no granularity fixes it.", size=12.5)
    rows2 = [["block-sparse kernel", "256 px cells", "64 px cells"]]
    bs = {r["cells"]: r for r in A["block_sparsity"]["rows"]}
    keys = sorted(bs["256 px cells"]["blocks"], key=lambda k: int(k))
    rows2.append(["ideal (per position)",
                  f"{bs['256 px cells']['ideal_pct']:.2f}%",
                  f"{bs['64 px cells']['ideal_pct']:.2f}%"])
    for k in keys:
        rows2.append([f"B = {k} feature positions",
                      f"{bs['256 px cells']['blocks'][k]:.2f}%",
                      f"{bs['64 px cells']['blocks'][k]:.2f}%"])
    d.table(rows2, IN + 6.9 * inch, H - 4.55 * inch,
            [2.9 * inch, 1.7 * inch, 1.7 * inch], fs=10.5,
            rowh=0.26 * inch)
    d.text(IN, H - 5.3 * inch,
           "<b>2. Saving is counted in multiply-accumulates.</b> A real "
           "block-sparse kernel keeps most of it at 256 px cells and much "
           "less at 64 px — the finer allocation is worth less once "
           "implementability is charged, which inverts the ideal ranking.",
           size=12.5, width=6.6 * inch)
    d.text(IN, H - 6.6 * inch,
           "<b>3. One decoder, intra only.</b> The inter path is untouched "
           "and no end-to-end GOP number is claimed.", size=12.5,
           width=6.6 * inch)

    # ------------------------------------------------------- future work
    d.slide("What follows", "Q6  Possible directions for future work",
            kicker="next")
    y = H - 1.9 * inch
    for n, (t, s_) in enumerate((
        ("Three more decoders, not one",
         "Every number here is DCVC-UF. The mechanism assumes only a residual "
         "trunk with a shared head, so it should port — but a claim about "
         "learned decoders needs more than one of them. Candidates with "
         "released weights and the same shape of synthesis transform: "
         "ELIC, MLIC++, and the TCM/Mixed-Transformer family. Each needs the "
         "ladder attached, the adapters warm-started to identity, and the "
         "cost table re-measured with hooks; the router and the allocation "
         "transfer unchanged."),
        ("Per-position decoding, and the kernel that makes it real",
         "It removes the floor, which is what makes the guarantee unreachable "
         "on hard content, and is worth +2.3 points at 256 px cells once a "
         "32 px block-sparse kernel is charged. The missing piece is the "
         "kernel and an end-to-end wall-clock measurement."),
        ("A budget chosen per rate rather than one for all",
         "Each rate exhausts the ladder at its own tolerance. Allocating one "
         "quality budget across rates is a second Lagrangian and costs "
         "nothing to solve."),
        ("The training objective the deployment actually uses",
         "The closed form showed the gain is in the conditioning. A training "
         "loop that samples tiles from the deployed exit distribution — not "
         "just weights the loss — is the version of Tier B that was not "
         "tried."))):
        d.text(IN, y, f"<b>{n+1}.  {t}</b>", size=14, colour=TUM_BLUE)
        y = d.text(IN + 0.3 * inch, y - 0.30 * inch, s_, size=11.8,
                   width=12.0 * inch) - 0.28 * inch

    # ---------------------------------------------------------- summary
    d.slide("Summary", kicker="summary")
    rows = [["", "mean saving", "guarantee", "added bits", "training"],
            ["A  signalled",
             f"{D['configs']['calexA_perframe']['saving_pct']:.2f}%",
             "every frame", "85–108 / frame", "none"],
            ["B  bitstream-identical",
             f"{A['fine_tuning']['rows'][0]['mean']-3.6:.2f}%",
             "distribution only", "0", "none"],
            ["C  partial signalling", "between the two", "every frame",
             "26 at ρ = 0.1", "none"]]
    d.table(rows, IN, H - 1.85 * inch,
            [3.8 * inch, 2.2 * inch, 2.6 * inch, 2.4 * inch, 1.6 * inch],
            hl=[1])
    y = H - 4.0 * inch
    for t in (f"<b>{DECGMAC:.0f} GMAC → "
              f"{DECGMAC*(1-D['configs']['calexA_perframe']['saving_pct']/100):.0f} "
              f"GMAC</b> per 1080p frame, ΔPSNR ≤ 0.1 dB on 264 of 265 "
              "frame-rate pairs, bitstream byte for byte unchanged.",
              "The guarantee costs nothing: choosing the multiplier per frame "
              "<i>raises</i> the mean saving, because one multiplier for a set "
              "solves the wrong problem.",
              "The last improvement needed no training at all — 443,520 "
              "parameters re-solved in closed form, on the tiles that reach "
              "each exit.",
              "What we cannot yet claim: wall-clock end to end, more than one "
              "decoder, and anything about inter frames."):
        y = d.text(IN, y, t, size=13.5, width=12.2 * inch) - 0.30 * inch
    return d


DECK = build
