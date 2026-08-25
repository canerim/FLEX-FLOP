"""One file with every figure, in reading order, each with what it shows.

Fifteen figures live in report/fig/ and are cited from two documents. Handing
someone that directory is handing them filenames; this puts them in the order
the argument runs, with a caption under each, as a single PDF and a single
tall PNG.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"

ORDER = [
    ("fig1_training", "The ladder the main experiment trains",
     "Six exits are trained; the deployed forward can select four. The clamp "
     "at j=2, and the two adapters the RD loss never reaches."),
    ("fig_configA", "Configuration A -- signalled",
     "The encoder searches the exact table and transmits the map. 27.64% mean "
     "at 0.1 dB, 64-91 bits/frame."),
    ("fig_configB", "Configuration B -- bitstream-identical",
     "Nothing added to the file. StemRouterHeadV2, 144k parameters, 0.163% of "
     "a decode. 24.69% mean. THIS is what the decoder side runs."),
    ("fig_configC", "Configuration C -- partial signalling",
     "The map is sent only where the predictor is worst. Beats A by 1.6 points "
     "at the lowest rate on the tiled ladder; does not transfer to 64 px / j=0."),
    ("fig2_mechanism", "Tiled decoding, per-position depth, and the band",
     "The cut and its seam; the dilation that replaces it; and the positions "
     "that keep computing for a deeper neighbour. The band is computed, not drawn."),
    ("fig10_pipeline", "The decode with the allocation path drawn in",
     "Every router input is already in the decoder, so nothing is added to the "
     "bitstream and the map is dilated once before the trunk runs."),
    ("fig11_budget", "Where one decode's compute goes",
     "The floor caps the ceiling: 0.609 tiled, 0.599 per-position, 0.301 once "
     "the clamp stops forcing four shared blocks on every position."),
    ("fig3_ladder", "The measured cost of each rung, and the spread by epoch",
     "What the clamp closes off, and the fact that the ladder's spread stops "
     "improving at the pinned epoch."),
    ("fig4_granularity", "Allocation cell against saving",
     "Finer cells buy saving and pay a dilation band; at 32 px the band eats "
     "the gain, which is why 64 px is the operating choice."),
    ("fig12_interaction", "The two levers are not additive",
     "Unlocking the clamp is worth +0.96 points at 256 px and +2.70 at 64 px; "
     "the gain is entirely at low rate."),
    ("fig5_frontier", "The compute-quality frontier",
     "Tiled against per-position at the extreme rates, with the interval the "
     "BD numbers integrate shaded."),
    ("fig6_waterfall", "Where the eight points come from",
     "Three inference-time levers applied in order to the pinned checkpoint. "
     "No weight retrained at any step."),
    ("fig7_gather", "Does the time follow the MAC count",
     "Gather, two pointwise convolutions, scatter, at the decoder's shapes. "
     "Time tracks the kept fraction to within 0.008-0.036, with RANDOM indices."),
    ("fig9_router", "The routing axis",
     "The error curve is nearly rank-1 at j=2, and the gap between oracle and "
     "a real router at both split depths."),
    ("fig13_surface", "Saving over rate and budget, and where it saturates",
     "Each rate meets the ceiling at its own budget -- 0.121 dB at q0 rising "
     "to 0.270 at q63 -- beyond which further tolerance buys nothing. At "
     "0.121317 dB exactly one rate is there and the rest are at 79-95% of it."),
    ("fig15a_frames_010", "0.10 dB: what the reported budget buys, frame by frame",
     "53 CTC frames at five rates, every point measured. The spread across "
     "content is wider than the spread across rates, and no rate yet rests "
     "on the ceiling."),
    ("fig15b_frames_perfect", "0.121317 dB: the perfect budget",
     "q0's whole row lies on the ceiling and nothing else does. This is the "
     "largest budget at which no tolerance is wasted: one rate has run out "
     "of depth to give up, the next does not until 0.153 dB."),
    ("fig15c_frames_020", "0.20 dB: past the point of return",
     "q0, q16 and q32 are pinned to the ceiling at 40.08%. The extra 0.079 dB "
     "over the perfect budget buys them nothing at all and buys q48 and q63 "
     "5.8 and 5.8 points."),
    ("fig8_epochs", "The paper's epoch series, independently confirmed",
     "Re-measured with current scripts on every surviving checkpoint."),
]


def build():
    n = len(ORDER)
    heights = []
    for name, _, _ in ORDER:
        im = mpimg.imread(FIG / f"{name}.png")
        heights.append(im.shape[0] / im.shape[1])
    W = 7.0
    pad = 0.78                      # inches reserved for the caption block
    fig_h = sum(h * W + pad for h in heights) + 0.9
    fig = plt.figure(figsize=(W, fig_h))
    fig.text(0.5, 1 - 0.30 / fig_h, "FLEX-UF  --  every figure, in the order "
             "the argument runs", ha="center", va="top", fontsize=11,
             fontweight="bold", color=S.INK)
    y = 1 - 0.80 / fig_h
    for i, ((name, title, cap), h) in enumerate(zip(ORDER, heights), 1):
        ih = h * W / fig_h
        fig.text(0.035, y, f"{i}.  {title}", fontsize=8.4, fontweight="bold",
                 va="top", color=S.INK)
        fig.text(0.965, y, f"fig/{name}.png", fontsize=6.2, va="top",
                 ha="right", color=S.MUTED, family="monospace")
        y -= 0.20 / fig_h
        ax = fig.add_axes([0.035, y - ih, 0.93, ih])
        ax.imshow(mpimg.imread(FIG / f"{name}.png"))
        ax.axis("off")
        y -= ih + 0.10 / fig_h
        # The caption gets the full width now; the filename rides on the
        # title line, where it cannot collide with a long sentence.
        import textwrap
        fig.text(0.035, y, "\n".join(textwrap.wrap(cap, 118)), fontsize=7.0,
                 va="top", color=S.INK2)
        y -= 0.34 / fig_h
    fig.savefig(HERE / "FLEX-UF-figures.pdf", dpi=200)
    fig.savefig(HERE / "FLEX-UF-figures.png", dpi=140)
    plt.close(fig)
    print(f"  {n} figur derlendi -> report/FLEX-UF-figures.pdf ve .png")


if __name__ == "__main__":
    build()
