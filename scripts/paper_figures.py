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
    "sys_pipeline.png",       # fig 1, the method
    "qualitative.png",        # what the saving looks like
    "adapters.png",           # exit adapters
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
    "latency.png",
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
