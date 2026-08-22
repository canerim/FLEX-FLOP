"""Every number in the paper must come from one set of weights.

The paper reports a pinned checkpoint. Nothing enforced that: after a repin,
the files the documents read are a mixture of the new checkpoint and whatever
each was last measured on, and a table drawn from epoch 4 can sit beside a
figure drawn from epoch 0 with nothing to say so. This is the check that
refuses that state.

Three kinds of file, and only the first has to move with a repin:

  weights     measured through our decoder. A repin invalidates it.
  structural  a property of the architecture or of the released codec, not of
              our weights -- MAC audits, tile geometry, latency, the bitrate
              of a bitstream we do not change. A repin leaves it alone.
  historical  deliberately measured on a named other checkpoint -- the second
              training run, an earlier epoch kept as a control, an ablation
              whose whole point is the checkpoint it was taken on.

A file is weights-class unless it is named here, so a new measurement is
guarded by default and someone has to think in order to exempt it.

    python scripts/check_epoch.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
PIN = R / "runs/RECIPE512/ckpt_PAPER.pth.tar"

# Structural: no dependence on which epoch of RECIPE512 produced them.
STRUCTURAL = {
    "adapter_cost.json": "MAC arithmetic for the adapter variants",
    "mac_audit.json": "hook count of the released decoder, by stage",
    "encoder_cost.json": "the encoder we never touch",
    "supp_encoder_cost_PAPER.json": "the encoder we never touch",
    "tile_definition.json": "tile geometry",
    "supp_module_shapes.json": "layer shapes",
    "supp_footprint.json": "parameter and activation footprint",
    "why_qp_PAPER.json": "bitrate of a bitstream we do not change",
    "rd_absolute_PAPER.json": "release PSNR and bitrate; our column is a check",
    "dmc_ld_recon_audit.json": "the released codec's own reconstruction",
    "seam_spatial_published.json": "geometry of the published seam",
    "logconvexity.json": "a property of the cost model",
    "hull_gap.json": "a property of the cost model",
    "theory_check.json": "propositions over the cost model",
    "theory_checks.json": "propositions over the cost model",
    "ctc_seam_p128.json": "tiling geometry",
    "ctc_seam_p256.json": "tiling geometry",
    "crosscheck_paths.json": "which file feeds which claim",
    "check_paper.json": "the checker's own output",
    "mac_crosscheck.json": "two MAC counters against each other",
    "design_space.json": "cost model enumeration",
}
# Latency and power: wall-clock of a fixed architecture. The exit mix at an
# operating point does move with the weights, so these are listed with the
# reason rather than waved through.
STRUCTURAL |= {n: "wall-clock of a fixed architecture" for n in (
    "latency_RECIPE512_sorted.json", "router_latency.json",
    "supp_latency_1920x1080.json", "supp_latency_1280x720.json",
    "supp_latency_batch_1920x1080.json", "supp_latency_cpu_1920x1080.json",
    "supp_power.json")}

# Derived: computed from other result files rather than from a decode, so
# they have no checkpoint of their own and asking for one is a category
# error. Their provenance is the provenance of what fed them, which is what
# check_provenance already follows.
DERIVED = {
    "band_collapse.json": "fitted to signalled_*_grid + saturation",
    "band_collapse_BEST.json": "the same fit on the second run",
    "bd_sensitivity.json": "BD integrals over the frontier",
    "bdrate.json": "paper_metrics, from the anchor and the sweeps",
    "exit_vs_rate.json": "summarised from the per-class histograms",
    "probe_rows.json": "a table assembled for the supplement",
    "router_inputs_rows.json": "a table assembled for the supplement",
    "router_retrain_compare.json": "two router files against each other",
    "setbudget_rows.json": "a table assembled for the supplement",
    "spread_stats.json": "per-sequence spread of the sweeps",
    "tilesize_adaptive.json": "arithmetic over the tile table",
    "frontier_law.json": "fitted to the frontier",
}

# Historical: measured on a named other checkpoint on purpose.
# Files whose whole point is a head fitted somewhere else. Both sides of the
# retrained-head comparison are heads from before the repin -- one before the
# exit-mask fix and one after -- and re-fitting either to the pinned weights
# would destroy the comparison rather than update it.
HEAD_HISTORICAL = {
    "hybrid_RECIPE512_b01_fixed.json": "the pre-fix head",
    "hybrid_v3_b01_pin.json": "the post-fix head",
}

HISTORICAL_PAT = [
    (re.compile(r"_BEST"), "the second training run"),
    (re.compile(r"BEST128"), "the 128 px run"),
    (re.compile(r"FINE12"), "the twelve-exit run"),
    (re.compile(r"VERBATIM"), "the verbatim-recipe run"),
    (re.compile(r"CONTROL"), "the control run"),
    (re.compile(r"_e\d+\.json$"), "an earlier epoch, kept as a control"),
    (re.compile(r"signalled_RECIPE512_\d{4}_\d{4}\.json$"),
     "the per-epoch watcher series"),
    (re.compile(r"^abandoned_"), "mechanisms measured and dropped"),
    (re.compile(r"before_pinning"), "kept to show what pinning changed"),
]


def _sources():
    return ([R / "scripts/build_pdf.py"]
            + sorted((R / "scripts/supp").glob("[a-z]_*.py"))
            + [R / "scripts/make_paper_tables.py",
               R / "scripts/paper_metrics.py"])


def _in_prose(text: str, pos: int) -> bool:
    """Is this offset inside a triple-quoted block or a comment?"""
    head = text[:pos]
    if head.count('"""') % 2 or head.count("'''") % 2:
        return True
    line = head.rsplit(chr(10), 1)[-1]
    return line.lstrip().startswith("#")


def fallbacks() -> set[str]:
    """Names a document mentions but does not read.

    Every reader here takes a list of candidates and uses the first that
    exists, so a document that names both the measurement on the pinned
    checkpoint and the older one it replaced reads only the first. Counting
    the second as stale would leave this check permanently red for files
    nothing reads any more, and deleting them would break a build that runs
    before the stage that writes the new one.
    """
    call = re.compile(r'(?:k\.J|self\.J|\bJ|_pick_json|pick|rows)\(\s*((?:f?"[^"]+\.json"'
                      r'(?:\s*,\s*)?)+)\s*[,)]', re.S)
    lst = re.compile(r'\[\s*((?:"[^"]+\.json"\s*,?\s*){2,})\]', re.S)
    # And as a bare tuple: the provenance table keeps its candidates as
    # (("a.json", "b.json"), "what it feeds").
    tup = re.compile(r'\(\s*((?:"[^"]+\.json"\s*,?\s*){2,})\)', re.S)
    behind, live = set(), set()
    for s in _sources():
        t = s.read_text()
        spans = []
        # A candidate list is sometimes a list rather than an argument
        # list: paper_metrics keeps ["a.json", "b.json", ...] beside the
        # budget it belongs to. Same meaning, same first-that-exists rule.
        for m in (list(call.finditer(t)) + list(lst.finditer(t))
                  + list(tup.finditer(t))):
            names = re.findall(r'"([A-Za-z0-9_.\-]+\.json)"', m.group(1))
            spans.append((m.start(1), m.end(1)))
            if len(names) < 2:
                live |= set(names)
                continue
            first = next((n for n in names if (R / "results" / n).exists()),
                         names[0])
            live.add(first)
            behind |= {n for n in names if n != first}
        # A name that also appears outside every candidate list is read on its
        # own somewhere, so it is not a fallback anywhere. Without this the
        # pre-fix hybrid file, which one section reads directly and another
        # lists behind the pinned-head measurement, was exempted from the
        # check that is there to catch exactly that file going stale.
        # Docstrings and comments do not count. Section F's docstring says
        # the curves "were first written as router_RECIPE512_b0*_PAPER.json
        # and then renamed", and reading that sentence as a live read put
        # three superseded files back into the sweep.
        for m in re.finditer(r'"([A-Za-z0-9_.\-]+\.json)"', t):
            if any(a <= m.start() < b for a, b in spans):
                continue
            if _in_prose(t, m.start()):
                continue
            live.add(m.group(1))
    return behind - live


def documents_read() -> set[str]:
    out = set()
    for s in _sources():
        t = s.read_text()
        out |= set(re.findall(r'"([A-Za-z0-9_.\-]+\.json)"', t))
        # A path literal with its directory in it -- R / "results/x.json" --
        # is the same read and was invisible to the pattern above, because a
        # slash is not in the character class. Two files feeding the
        # qualitative macros were being read that way and never checked.
        out |= {m.rsplit("/", 1)[-1] for m in
                re.findall(r'"(?:results|\./results)/([A-Za-z0-9_.\-]+\.json)"', t)}
    return out - fallbacks()


def main() -> int:
    import torch
    want = None
    if PIN.exists():
        want = torch.load(PIN, map_location="cpu",
                          weights_only=False).get("epoch")
    if want is None:
        print("  no pinned checkpoint to compare against")
        return 1

    stale, unlabelled, ok, skipped = [], [], 0, 0
    # A file can be measured on the pinned decoder and still report a
    # configuration the paper does not: the router head is a second set of
    # weights, and one fitted to a different epoch answers a question about
    # that epoch. Twice today a file like that was the newest on disk and won
    # the tie-break in both readers.
    head_off = []
    for n in sorted(documents_read()):
        p = R / "results" / n
        if not p.exists():
            continue
        if n in STRUCTURAL or n in DERIVED:
            skipped += 1
            continue
        if any(pat.search(n) for pat, _ in HISTORICAL_PAT):
            skipped += 1
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ck = str(d.get("ckpt", ""))
        # ckpt_eval is the file the watcher overwrites every epoch, so a
        # result that names it names nothing: it cannot be checked, only
        # re-measured.
        if "ckpt_eval" in ck or "ckpt_step" in ck:
            unlabelled.append(f"{n}  (names a moving checkpoint)")
            continue
        if "RECIPE512" not in ck:
            # No checkpoint recorded and not exempt: nobody can tell what it
            # was measured on, which is the same problem in a quieter form.
            if "ckpt" not in d:
                unlabelled.append(n)
            continue
        e = d.get("ckpt_epoch")
        if e is None:
            unlabelled.append(n)
        elif e != want:
            stale.append((n, e))
        else:
            meta = d.get("router2_meta") or {}
            # Compare what the head was fitted to against what the file was
            # measured on. The epoch is the wrong field for this: the heads
            # fitted before the repin record no epoch at all, and "None !=
            # 4" reads as a different epoch when what it means is a head
            # fitted to a checkpoint that moves.
            hck = str(meta.get("ckpt") or "")
            # A head that is named but does not run is not part of the
            # measurement: the rate-rank hybrid passes one and reports a
            # compute share of zero because the base predictor is the rule.
            _unused = d.get("router_compute_share_pct") == 0
            if (d.get("router2") and hck and hck != ck and not _unused
                    and n not in HEAD_HISTORICAL):
                head_off.append((n, hck.rsplit("/", 1)[-1]))
            else:
                ok += 1

    for n, e in stale:
        print(f"     epoch {e}, wanted {want}: {n}")
    for n in unlabelled:
        print(f"     no epoch recorded: {n}")
    for n, he in head_off:
        print(f"     measured on the pin, with a head fitted to {he}: {n}")
    print(f"\n  {ok} on epoch {want}, {len(stale)} stale, "
          f"{len(unlabelled)} unlabelled, {len(head_off)} with a head off the "
          f"pin, {skipped} exempt")
    return 1 if (stale or unlabelled or head_off) else 0


if __name__ == "__main__":
    sys.exit(main())
