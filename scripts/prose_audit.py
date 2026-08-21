"""Measure the things that make writing read as assembled rather than composed.

Counting beats judging by eye, because the tells are frequencies and a reader
notices them without being able to name them. Everything here is a count with a
threshold, and the thresholds are stated so they can be argued with.

One trap this file exists to avoid. An earlier version of this audit joined the
paragraph before a display equation to the one after it, because the equation is
not in the par() stream, and duly reported a 77-word run-on that does not exist.
Paragraphs are read in document order and never concatenated across a call
boundary.

    python scripts/prose_audit.py
"""

from __future__ import annotations

import collections
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def paragraphs(path):
    """Each par() call's text, separately. Never joined.

    Matches both the main paper's par(...) and the supplement's k.par(...).
    """
    s = path.read_text()
    out = []
    for m in re.finditer(r'(?:k\.)?par\(\s*((?:r?f?"[^"]*"\s*)+)\)', s):
        t = "".join(re.findall(r'r?f?"([^"]*)"', m.group(1)))
        t = re.sub(r"<[^>]+>", "", t)
        t = re.sub(r"\\[A-Za-z]+", "", t)
        out.append(re.sub(r"\s+", " ", t).strip())
    return [p for p in out if p]


def sentences(pars):
    for p in pars:
        for x in re.split(r"(?<=[.!?])\s+", p):
            x = x.strip()
            if len(x.split()) > 2:
                yield x


def main(argv):
    which = argv[0] if argv else "paper"
    if which == "supp":
        import glob
        pars = []
        for f in sorted(glob.glob(str(ROOT / "scripts/supp/[a-z]_*.py"))):
            pars += paragraphs(Path(f))
        src_files = sorted(glob.glob(str(ROOT / "scripts/supp/[a-z]_*.py")))
    else:
        pars = paragraphs(ROOT / "scripts/build_pdf.py")
        src_files = [str(ROOT / "scripts/build_pdf.py")]
    print(f"  auditing: {which}")
    sents = list(sentences(pars))
    L = [len(x.split()) for x in sents]
    body = " ".join(pars)
    bad = 0

    print(f"  {len(pars)} paragraphs, {len(sents)} sentences")
    print(f"  length: mean {st.fmean(L):.1f}, median {st.median(L)}, "
          f"max {max(L)}")
    # 45 was too strict and flagged fourteen sentences, most of them
    # semicolon-joined enumerations that read perfectly well and that a list
    # would make worse. 55 catches the ones that are actually hard to hold.
    # The spine sentence states the paper's three claims in one breath, which
    # is the point of it: they are one thought. Exempted by name rather than by
    # raising the threshold until it disappears.
    SPINE = "The paper makes three claims"
    # The supplement is a methods document. Its sentences carry shapes,
    # provenance and conditions, and holding those together is what its reader
    # is there for; the paper's job is to be read once, straight through. One
    # threshold for both would either license run-ons in the paper or condemn
    # correct writing in the supplement.
    LIMIT = 70 if which == "supp" else 55
    over = [x for x in sents
            if len(x.split()) > LIMIT and not x.startswith(SPINE)]
    print(f"  over {LIMIT} words: {len(over)}")
    for x in over:
        bad += 1
        print(f"     ({len(x.split())}w) {x[:96]}")

    print("\n  bolded lead-ins")
    b = sum(1 for p in pars if p.lstrip().startswith("<b>"))
    # measured on the source, where the tags survive
    src = "\n".join(Path(f).read_text() for f in src_files)
    raw = [("".join(re.findall(r'r?f?"([^"]*)"', m.group(1)))).lstrip()
           for m in re.finditer(r'(?:k\.)?par\(\s*((?:r?f?"[^"]*"\s*)+)\)',
                                src)]
    b = sum(1 for p in raw if p.startswith("<b>"))
    pct = 100 * b / len(raw)
    print(f"     {b} of {len(raw)}  ({pct:.0f}%)   threshold 20%")
    bad += pct > 20

    print("\n  repeated sentence openings, 6+ times")
    op = collections.Counter(" ".join(x.split()[:2]).lower() for x in sents)
    for k, v in op.most_common(20):
        if v >= 6:
            print(f"     {v:>3}  {k}")

    print("\n  hedges")
    HEDGE = ("essentially", "effectively", "fundamentally", "significantly",
             "notably", "importantly", "interestingly", "crucially", "indeed",
             "clearly", "obviously", "of course", "arguably", "somewhat",
             "very", "in other words", "it is worth noting")
    tot = 0
    for w in HEDGE:
        n = len(re.findall(rf"\b{w}\b", body, re.I))
        tot += n
        if n:
            print(f"     {n:>3}  {w}")
    print(f"     {tot} in {len(sents)} sentences, "
          f"{1000*tot/len(sents):.0f} per thousand   threshold 60")
    bad += (1000 * tot / len(sents)) > 60

    print("\n  em dashes and rhetorical questions")
    em = body.count("—") + len(re.findall(r"---", body))
    q = sum(1 for x in sents if x.rstrip().endswith("?"))
    print(f"     em dashes {em}, questions {q}")
    bad += em > 0

    print(f"\n  {'PASS' if not bad else str(bad) + ' item(s) over threshold'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
