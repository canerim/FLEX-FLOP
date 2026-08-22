"""Every number the abstract states, with the file it came from.

The abstract is the part a reader believes without checking, and every number
in it is a macro, so it moves silently when the tables are regenerated. That
is the design working -- and it also means nobody sees it happen. This prints
the macros the abstract uses, their current values, and which result file each
was computed from, so a repin can be checked at the place it matters most.

    python scripts/check_abstract.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent

# Where each abstract macro is computed. Not guessed: grepped from
# make_paper_tables.py and paper_metrics.py, and wrong here is a failure.
SOURCE = {
    "DecGmac": "mac_audit.json",
    "NumSeq": "signalled_RECIPE512_ctc53.json",
    "MainLowRate": "signalled_RECIPE512_ctc53.json",
    "MainHighRate": "signalled_RECIPE512_ctc53.json",
    "MeanAtOne": "signalled_RECIPE512_ctc53.json",
    "BdRateALow": "bdrate.json",
    "BandRawSpread": "band_collapse.json",
    "BandSpreadMean": "band_collapse.json",
    "RouterParams": "supp_module_shapes.json",
    "RateRankBeatsBy": "raterank_RECIPE512_b01.json",
}


def macros():
    t = (R / "paper/tables/macros.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", t))


def main() -> int:
    m = macros()
    used = sorted(set(re.findall(
        r"\\\\([A-Z][A-Za-z]+)",
        (R / "scripts/build_pdf.py").read_text()[
            (R / "scripts/build_pdf.py").read_text().find("---- abstract"):][:3000])))
    # The tex twin is the authority on what the abstract actually says.
    tex = (R / "paper/main.tex").read_text()
    a0, a1 = tex.find("\\begin{abstract}"), tex.find("\\end{abstract}")
    used = sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", tex[a0:a1])))

    bad = 0
    print(f"  {'macro':<18}{'value':>12}   epoch  source")
    for name in used:
        val = m.get(name)
        src = SOURCE.get(name, "?")
        ep = "-"
        p = R / "results" / src
        if p.exists():
            try:
                d = json.loads(p.read_text())
                ep = d.get("ckpt_epoch", "-") if isinstance(d, dict) else "-"
            except Exception:
                pass
        flag = ""
        if val is None:
            flag, bad = "   UNDEFINED", bad + 1
        elif src == "?":
            flag, bad = "   no source recorded", bad + 1
        print(f"  \\{name:<17}{val or '--':>12}   {str(ep):>5}  {src}{flag}")
    print(f"\n  {len(used)} macros in the abstract, {bad} without a value or "
          f"a source")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
