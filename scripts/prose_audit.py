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
        # The supplement builds its numbers with f-strings, so a paragraph's
        # source carries Python inside {...}: min(fl(f)), a comprehension, a
        # format spec. Left in, that code is counted as words -- it inflated
        # sentence lengths and produced 43 "repeated phrases" that were all the
        # same expression appearing twice. Substitute a placeholder rather than
        # deleting, so the sentence keeps a subject where a number will stand.
        t = re.sub(r"\{[^{}]*\}", "0", t)
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

    # A phrase repeated inside one paragraph is the clearest signature of text
    # that was generated and not read back. Two got past every other check:
    # section 5.1 opened "A tenth of a decibel is an engineering convention. It
    # is an engineering convention, chosen because...", and contribution (iii)
    # stated its result and then stated it again in different words.
    #
    # Six words, not four. At four the check fires 19 times on this paper and
    # every one is a technical phrase the subject genuinely repeats -- "the
    # calibrated bit rule", "each rate's own band". At six, a repeat inside one
    # paragraph is an accident, and both real defects were six-word repeats.
    print("\n  phrases repeated inside one paragraph")
    N = 6
    rep = []
    for p_ in pars:
        w = re.findall(r"[a-z]+", p_.lower())
        seen = collections.Counter(" ".join(w[i:i + N])
                                   for i in range(len(w) - N + 1))
        for g, n in seen.items():
            # Stripped mathematics leaves runs of single letters: the proof of
            # (A1) has "D D'' - (D')^2", which reduces to six letter-ds and is
            # not a repeated phrase. Require six real words carrying at least
            # three distinct ones.
            if n > 1 and len({x for x in g.split() if len(x) > 2}) >= 3:
                rep.append((n, g, p_[:70]))
    for n, g, where in rep[:8]:
        print(f"     {n}x  \"{g}\"  in: {where}...")
    print(f"     {len(rep)} repeated {N}-grams   "
          f"threshold {'report only' if which == 'supp' else 0}")
    # The paper is read once, straight through, and a phrase arriving twice in
    # one paragraph reads as a fault there. The supplement is a methods
    # document whose paragraphs compare the same quantity across rates, and
    # "at q0 the smaller tile is ahead ... at q63 the smaller tile is behind"
    # is the sentence doing its job. Reported in both, fatal only in the paper.
    if which != "supp":
        bad += len(rep) > 0

    print("\n  em dashes and rhetorical questions")
    em = body.count("—") + len(re.findall(r"---", body))
    q = sum(1 for x in sents if x.rstrip().endswith("?"))
    print(f"     em dashes {em}, questions {q}")
    bad += em > 0

    print(f"\n  {'PASS' if not bad else str(bad) + ' item(s) over threshold'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
