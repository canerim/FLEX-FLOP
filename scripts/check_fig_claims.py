"""A reference to a figure must sit in a sentence that says something.

check_fig_prose already insists that every figure is read somewhere -- caption
or citing paragraph. This is the next rule up, and it is the one CVPR and
NeurIPS reviewers apply without being asked: the sentence carrying the
reference should state the finding, not announce that a picture exists.

    weak    The trade-off is plotted in Figure 12.
    strong  At the lowest rate 0.1 dB buys 29.2% and at the highest 15.2%
            (Figure 12).

A sentence is accepted when it carries a number, a comparison, or a verb that
asserts something about the data. It is flagged when it is only a pointer --
"see", "is shown in", "we plot", "presents the results" -- with nothing beside
it.

    python scripts/check_fig_claims.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
LIT = re.compile(r'r?f?"((?:[^"\\]|\\.)*)"')
REF = re.compile(r"\[\[(?:fig|tab):[a-z0-9_]+\]\]|"
                 r"\b(?:Figure|Fig\.|Table)~?\s*\d+", re.I)
# Enough to be a claim: a measurement, a comparison, or an assertion about it.
SUBSTANCE = re.compile(
    r"\d|\bmore\b|\bless\b|\bhigher\b|\blower\b|\bbeats?\b|\bthan\b|\bfalls?\b|"
    r"\brises?\b|\bworse\b|\bbetter\b|\bnarrow(?:s|er)?\b|\bwiden(?:s|er)?\b|"
    r"\bflat(?:tens)?\b|\bcollapse[sd]?\b|\bagree(?:s|ment)?\b|\bcross(?:es)?\b|"
    r"\bevery\b|\bno\b|\bnone\b|\bonly\b|\bmost\b|\bhalf\b|\bdouble[sd]?\b|"
    r"\bgrows?\b|\bshrinks?\b|\bcosts?\b|\bsaves?\b|\bpredicts?\b|\bdominates?\b|"
    r"\bfollows?\b|\btracks?\b|\bwide[rn]?\b|\bnarrow\b|\bnot\b|\bapart\b|"
    r"\bdiffers?\b|\bsame\b|\bbetween\b|\bbelow\b|\babove\b|\binside\b|"
    r"\boutside\b|\bnever\b|\balways\b|\bfirst\b|\blast\b|\bwins?\b|"
    r"\bloses?\b|\bmisses?\b|\bhides?\b|\bexplains?\b",
    re.I)
# Pure pointers, when they are all a sentence has.
POINTER = re.compile(
    r"\bsee\b|\bis (?:shown|plotted|given|presented|drawn|reported)\b|"
    r"\bwe (?:plot|show|present|report|give)\b|\bshows the\b|"
    r"\bpresents the\b|\bsummari[sz]es the\b|\bare (?:shown|plotted|given)\b",
    re.I)


def prose(path: Path) -> list[tuple[int, str]]:
    src = path.read_text()
    out = []
    for m in re.finditer(r"\n\s*(?:par|h2|note)\(", src):
        seg, d, e = src[m.end():m.end() + 3500], 1, None
        for i, c in enumerate(seg):
            if c == "(":
                d += 1
            elif c == ")":
                d -= 1
                if d == 0:
                    e = i
                    break
        if e is None:
            continue
        t = "".join(x.group(1) for x in LIT.finditer(seg[:e]))
        t = re.sub(r"<[^>]+>", "", t)
        out.append((src[:m.end()].count("\n") + 1, re.sub(r"\s+", " ", t)))
    return out


def main() -> int:
    weak = []
    for f in [R / "scripts/build_pdf.py"] + sorted((R / "scripts/supp").glob("*.py")):
        for line, para in prose(f):
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if not REF.search(sent):
                    continue
                # the reference itself is not substance
                bare = REF.sub(" ", sent)
                if SUBSTANCE.search(bare):
                    continue
                if POINTER.search(sent) or len(bare.split()) < 9:
                    weak.append((f.name, line, sent.strip()[:96]))
    for f, line, s in weak:
        print(f"     {f}:{line} points without a finding: {s!r}")
    print(f"\n  {'PASS' if not weak else str(len(weak)) + ' bare pointer(s)'}")
    return 1 if weak else 0


if __name__ == "__main__":
    sys.exit(main())
