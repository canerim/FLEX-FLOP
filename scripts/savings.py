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


def pick(*names, res=None):
    """The canonical result file among candidates, by provenance not by order.

    Same rule as make_paper_tables.pick, and here so a figure script cannot
    disagree with the table beside it. raterank_figure.py hardcoded
    router_RECIPE512_b01_fixed.json and drew a "144 K head" curve at 27.2% at
    q0 while the table printed 23.6: a different head, measured before the
    checkpoint was pinned.

    Prefer a file measured on the pinned checkpoint, then the most recently
    written. The order the names are given in only breaks ties.
    """
    import json as _json
    from pathlib import Path as _Path
    res = _Path(res) if res else _Path(__file__).resolve().parent.parent / "results"
    found = [(n, res / n) for n in names if (res / n).exists()]
    if not found:
        return None, None

    def rank(item):
        n, p = item
        pinned = 0
        try:
            d = _json.load(open(p))
            pinned = 1 if "ckpt_PAPER" in str(d.get("ckpt", "")) else 0
        except Exception:
            pass
        return (-pinned, -p.stat().st_mtime)

    n, p = sorted(found, key=rank)[0]
    return _json.load(open(p)), n


def epoch_series(run="RECIPE512", n_seq=53, budget=0.1, res=None):
    """{epoch: mean saving} for one run, on the frames the paper reports.

    Three rules, and each was got wrong somewhere before it was written down.

    Only the full test set. The set grew from 40 sequences to 53 when MCL-JCV
    and the two smallest HEVC classes were added, and small frames save far
    less, so a 40-sequence number sits several points above a 53-sequence one
    at the same weights.

    Only the hook count. sv() falls back to the arithmetic model when a file
    carries no hook count, silently and in our favour, and one epoch-1 file
    has no hook count: a series built with the fallback read 24.1% at epoch 1
    where the hook-counted file reads 22.8%, which is two conventions averaged
    into one curve.

    The median where an epoch was measured more than once, never the best.
    Epoch 0 has four files; max() over them would manufacture a rising series
    the moment they disagreed.

        from savings import epoch_series
        epoch_series()          # {0: 21.5, 1: 22.8, ...}
    """
    import glob as _glob
    import json as _json
    import statistics as _st
    from pathlib import Path as _Path
    res = _Path(res) if res else _Path(__file__).resolve().parent.parent / "results"
    ser = {}
    for f in _glob.glob(str(res / f"signalled_{run}_*.json")):
        try:
            d = _json.loads(_Path(f).read_text())
        except Exception:
            continue
        if d.get("n_sequences") != n_seq:
            continue
        rows = [r for r in d.get("rows", [])
                if r.get("budget_db") == budget and r.get("budget_reachable")]
        if not rows or not measured(rows[0]):
            continue
        vals = [v for v in (sv(r) for r in rows) if v is not None]
        if len(vals) < 5:
            continue
        e = d.get("ckpt_epoch")
        if e is not None:
            ser.setdefault(e, []).append(sum(vals) / len(vals))
    return {e: _st.median(v) for e, v in sorted(ser.items())}
