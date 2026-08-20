"""Copy the figures the paper uses out of docs/figures into paper/figures.

A copy rather than a symlink: the paper directory has to survive being zipped
and sent to a submission system, and a dangling link is a silent failure that
surfaces as a missing figure in the compiled PDF.
"""
import shutil
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SRC, DST = R / "docs" / "figures", R / "paper" / "figures"
DST.mkdir(parents=True, exist_ok=True)

WANTED = [
    "sys_pipeline.png",
    "dcvcuf_framework.png",
    "field.png",             # what every published decoder costs
    "tiles_unequal.png",     # why one depth per frame is the wrong shape
    "patchify.png",          # how a frame becomes 40 independent tiles
    "ladder.png",            # the per-exit cost, measured against modelled
    "allocation.png",        # where the tiles go, by rate
    "router_arch.png",       # the router head, layer by layer   # the baseline, from the DCVC-UF paper
    "baseline.png",           # the decoder we modify, for the introduction       # fig 1, the method
    "qualitative.png",        # what the saving looks like
    "adapters.png",           # exit adapters
    "adapter_gain.png",       # what the adapters are worth
    "seam_problem.png",       # the tiling artefact at frame scale
    "seam_module.png",        # grid seam repair and where it acts
    "seam_vs_qp.png",         # seam against rate
    "saturation_RECIPE512.png",   # the operating-point structure
    "headline_RECIPE512.png",     # main result
    "router_ab.png",          # signalled vs predicted
    "training_scheme.png",    # objective and recipe
    "exit_map.png",           # per-tile exit map on a real frame
    "perclass.png",           # saving by test class vs tile count
    "contamination.png",      # seam against per-tile depth
    "map_transfer.png",       # reusing a map across time and rate
    "rd_spread.png",          # rate-quality plane and per-sequence spread
    "theory.png",             # the structure of the allocation
    "latency.png",
    "hybrid.png",             # partial signalling: A and B are one scale
    "raterank.png",           # the parameter-free control
    "blend.png",              # do the two decoder-side signals differ?
    "tradeoff.png",           # the whole budget-to-saving curve
    "budget_band.png",        # the same, normalised onto each rate's band
]

missing = []
for f in WANTED:
    s = SRC / f
    if s.exists():
        shutil.copy2(s, DST / f)
        print(f"  {f}")
    else:
        missing.append(f)
if missing:
    print("\n  not yet rendered: " + ", ".join(missing))
