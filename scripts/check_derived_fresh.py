"""A file computed from another file must be newer than it.

check_figs_fresh does this for figures. Nothing did it for the results
files that are arithmetic over other results files, and the exemption that
lets them skip the epoch check -- "their provenance is the provenance of
what fed them" -- is only true while they have actually been recomputed.

Four of them had not been. The Pareto enumeration, the log-convexity check
and the tile-size splice were all computed over results/tile_table.json in
August; the table was re-measured on the pinned checkpoint months of
commits later and the three numbers derived from it did not move. One of
them was a proposition's headroom, and it was wrong by a factor of three.

For each derived file this finds the script that writes it, the results
files that script reads, and compares the modification times.

    python scripts/check_derived_fresh.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
RES = R / "results"
SKIP = {"check_derived_fresh.py", "check_epoch.py", "check_paper.py",
        "check_provenance.py", "check_numbers.py", "check_figs_fresh.py"}


def derived_names() -> dict[str, str]:
    """The files check_epoch waves through as derived, and why."""
    t = (R / "scripts/check_epoch.py").read_text()
    m = re.search(r"DERIVED = \{(.*?)\n\}", t, re.S)
    if not m:
        return {}
    return dict(re.findall(r'"([^"]+\.json)":\s*"([^"]*)"', m.group(1)))


# What each derived file is computed from, written down rather than guessed.
# Scanning the producing script for .json literals does not work: the script
# that writes most of these reads sixty other files for other reasons, so
# everything it writes looks stale. Someone has to think once, here.
# Derived by check_epoch's reckoning, but not arithmetic over anything
# current: the record of a head before an exit-mask fix and the same head
# after, both measured before the repin. Re-deriving it against a head fitted
# to the pinned weights would destroy the comparison rather than refresh it.
HISTORICAL = {"router_retrain_compare.json"}

INPUTS = {
    "band_collapse.json": ["signalled_RECIPE512_grid.json",
                           "saturation_RECIPE512_ctc53.json"],
    "band_collapse_BEST.json": ["signalled_BEST_grid.json",
                                "saturation_BEST_ctc53.json"],
    "bd_sensitivity.json": ["supp_paper_curve_PAPER.json"],
    "bdrate.json": ["signalled_RECIPE512_ctc53.json",
                    "rd_absolute_PAPER.json"],
    "exit_vs_rate.json": ["per_class_RECIPE512.json"],
    "frontier_law.json": ["supp_paper_curve_PAPER.json"],
    "hull_gap.json": ["tile_table.json"],
    "logconvexity.json": ["tile_table.json"],
    "probe_rows.json": ["supp_opquality_PAPER.json"],
    "router_inputs_rows.json": ["router_ablation_e4.json"],
    # Not listed: results/router_retrain_compare.json. No script writes
    # it, and it is not arithmetic over anything current -- it is the
    # record of a head before an exit-mask fix and the same head after,
    # both measured before the repin, which is the comparison it exists
    # to be. Re-deriving it against a head fitted to the pinned weights
    # would destroy it rather than refresh it.
    "setbudget_rows.json": ["signalled_RECIPE512_grid.json"],
    "spread_stats.json": ["supp_per_sequence_PAPER_b010.json"],
    "rd_spread_stats.json": ["supp_per_sequence_PAPER_b010.json"],
    "tilesize_adaptive.json": ["tile_table.json", "per_class_RECIPE512.json"],
    "theory_check.json": ["tile_table.json"],
    "theory_checks.json": ["tile_table.json"],
}


def main() -> int:
    stale, checked, unlisted = [], 0, []
    for name in sorted(derived_names()):
        p = RES / name
        if not p.exists():
            continue
        srcs = INPUTS.get(name)
        if srcs is None:
            if name in HISTORICAL:
                continue
            unlisted.append(name)
            continue
        checked += 1
        mine = p.stat().st_mtime
        newer = sorted(n for n in srcs
                       if (RES / n).exists()
                       and (RES / n).stat().st_mtime > mine)
        if newer:
            stale.append((name, newer))
    for name, newer in stale:
        print(f"     {name} is older than what it is computed from: "
              f"{', '.join(newer)}")
    for name in unlisted:
        print(f"     {name}: nothing here says what it is computed from")
    print(f"\n  {checked} derived file(s) checked, {len(stale)} behind their "
          f"inputs, {len(unlisted)} unlisted")
    return 1 if (stale or unlisted) else 0


if __name__ == "__main__":
    sys.exit(main())
