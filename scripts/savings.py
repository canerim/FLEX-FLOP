"""One definition of "saving", for everything that reports one.

There are two in the result files. `saving_pct_vs_release` is the arithmetic
model: exit costs summed from the ladder's own table. `saving_pct_measured` is
what a hook on every convolution counted during the decode that produced the
number. They disagree, always in our favour, because the model under-bills the
shallow exits by a constant 0.008 of a released decode -- 0.4 to 0.8 saving
points, and up to 3.3 where the exit mix is extreme.

The paper reports the hook count. make_paper_tables and check_paper have used
this rule for a while; the figure scripts did not, so `rd_spread.png` printed
"0.1 dB, 24% saved" beside a table that said 21.5, and its per-rate labels ran
0.7 points high. A number drawn inside a PNG is invisible to every checker in
this repo, which makes it exactly the number that must not be computed twice.

    from savings import sv
    sv(row)                      # the deployed saving, hook-counted
    sv(row, "oracle_saving_pct") # the same for the Lagrangian oracle
"""

from __future__ import annotations


def sv(row, key="saving_pct"):
    """The saving a row reports, measured off the decode where available."""
    m = row.get(key + "_measured")
    return m if m is not None else row.get(key + "_vs_release")


def sv_oracle(row):
    """The Lagrangian oracle's saving on the same row, same definition."""
    return sv(row, "oracle_saving_pct")


def measured(row, key="saving_pct"):
    """True when the row carries a hook count, so a table can mark the rest."""
    return row.get(key + "_measured") is not None
