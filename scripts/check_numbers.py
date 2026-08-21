"""Every decimal number printed in the prose must come from somewhere.

The paper's protocol paragraph carried two numbers -- "3.29 saving points" and
"7 to 10" -- that are in no results file, no macro and no checked claim. One of
them was off by a factor of three against what the pinned files measure. They
survived because nothing connects a number typed into a sentence to the
measurement it came from.

A decimal literal in prose is accounted for if it is the value of a macro, or
the expected side of a check_paper claim, or on the small list below of
constants the paper states rather than measures.

    python scripts/check_numbers.py
"""
from __future__ import annotations

import io
import os
import re
import sys
import tokenize
from pathlib import Path

R = Path(__file__).resolve().parent.parent

# Stated, not measured: the budget grid, section numbers, and the page
# geometry in build_pdf's own layout code.
# Stated rather than measured, each with the reason it is here. Printed on
# every run, so nothing is excused quietly.
ALLOW = {
    "0.1": "the headline budget", "0.2": "budget grid", "0.25": "budget grid",
    "0.3": "budget grid", "0.5": "budget grid", "0.05": "budget grid",
    "0.15": "budget grid", "0.75": "blend weight", "0.9": "blend weight",
    "1.0": "unit", "0.0": "unit", "2.0": "unit", "1.5": "unit",
    "0.4": "blend weight", "0.6": "unit", "0.8": "unit",
    # Section 5.6's account of the exit-mask bug describes measurements that
    # were taken before it was fixed. They are history, they are labelled as
    # history in the sentence itself, and no file holds them any more.
    "1.7": "pre-fix exit-mask gap, historical",
    "13.3": "pre-fix exit-mask gap, historical",
    "3.5": "cost of fixing the exit mask, historical",
    "9.4": "gain from fixing the exit mask, historical",
}
# A section reference, recognised by what precedes it rather than by its
# shape: "^\\d\\.\\d{1,2}$" also matches 4.5, 2.38 and 7.77, which is most of
# the numbers this check exists to look at.
SECTION_BEFORE = re.compile(
    r"(?:sections?|§|appendix|figs?\.|figures?|tables?|eq\.|equations?)\s+$",
    re.I)


def prose_numbers(path: Path) -> dict[str, list[int]]:
    """Numbers in what the paper prints, not in what builds it.

    Only the strings passed to par(), h2(), figure() and tbl() are read. An
    earlier version tokenised every string in the file and reported the column
    widths in build_pdf's own layout docstring.
    """
    src = path.read_text()
    out: dict[str, list[int]] = {}
    lit = re.compile(r'r?f?"((?:[^"\\]|\\.)*)"')
    # build_pdf writes `par(...)`; the supplement sections write `k.par(...)`
    # on a Kit object. Matching only the bare form made the supplement half of
    # this check a no-op that reported PASS over zero sentences.
    for m in re.finditer(r"\n\s*(?:story \+= )?(?:k\.)?(?:par|h1|h2|note|fig"
                         r"|figwide|figure_wide|figure|tbl|rows)\(", src):
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
            # An unbalanced slice means the call ran past the 3500-character
            # window; reading to the end of it swept in the layout code after
            # the paragraph and reported a font size as a claim.
            continue
        line = src[:m.end()].count("\n") + 1
        text = "".join(x.group(1) for x in lit.finditer(seg[:e]))
        for mm in re.finditer(r"(?<![\d.\w-])(\d{1,4}\.\d{1,4})(?![\d.])",
                              text):
            if SECTION_BEFORE.search(text[max(0, mm.start() - 12):mm.start()]):
                continue
            out.setdefault(mm.group(1), []).append(line)
    return out


def section_numbers() -> set[str]:
    """The subsection numbers the paper actually has, from its own h2 calls.

    A cross-reference split across two string literals loses the word "Section"
    that precedes it, so the number arrives bare. Allowing every d.dd would
    excuse 4.5 and 2.38 as well; allowing only the numbers that are real
    subsection headings excuses exactly the references.
    """
    src = (R / "scripts/build_pdf.py").read_text()
    return set(re.findall(r'h2\("(\d+\.\d+)', src))


def measured() -> set[str]:
    """Every number in results/, in the forms a paper prints it in.

    The supplement computes almost all of its numbers in f-strings, so a typed
    decimal there is the exception and worth checking. Requiring it to be a
    macro or a claim would be too strong -- the supplement legitimately quotes
    values no macro carries -- so the test is that it exists SOMEWHERE in the
    measurements.

    Be clear about what that does and does not catch. Fifty thousand printed
    forms come out of results/, so almost any two-decimal number is in the
    index and a value that drifted from 0.25 to 0.27 will pass. What it does
    catch is a number that exists nowhere in the measurements at all -- which
    is precisely how "3.29 saving points" and "7 to 10" survived in the main
    paper's protocol paragraph, one of them wrong by a factor of three. An
    orphan, not a drift.

    Mantissas are indexed too. Lambdas are stored as 5.126e-05 and printed as
    "5.126 x 10^-5", and without the mantissa form every one of them reads as
    unbacked.
    """
    import glob
    import json as _j
    out: set[str] = set()

    def add_num(o):
        a = abs(o)
        for d in (1, 2, 3, 4):
            out.add(f"{a:.{d}f}")
            out.add(f"{a:.{d}f}".rstrip("0").rstrip(".") or "0")
        if 0 < a < 0.01 or a >= 1e4:
            m = f"{a:e}".split("e")[0]
            for d in (1, 2, 3, 4):
                out.add(f"{float(m):.{d}f}")
                out.add(f"{float(m):.{d}f}".rstrip("0").rstrip(".") or "0")

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            add_num(o)

    for f in sorted(glob.glob(str(R / "results/*.json"))):
        try:
            if os.path.getsize(f) > 20_000_000:
                continue
            walk(_j.loads(open(f).read()))
        except Exception:
            pass
    return out


def known() -> set[str]:
    """Every number the build can justify: macro values, and the expected side
    of each check_paper claim.

    The claims are read out of check_paper's source rather than by running it.
    Running it made this check quadratic and then recursive, because
    check_paper runs this one.
    """
    vals = set(re.findall(r"\\newcommand\{\\[A-Za-z]+\}\{([^}]*)\}",
                          (R / "paper/tables/macros.tex").read_text()))
    got: set[str] = set()
    for v in vals:
        got |= set(re.findall(r"-?\d+\.?\d*", v))
    # Every literal in check_paper, not only the second argument of claim().
    # The seam exponents are asserted from a tuple -- ("0", 2.38), ("32", 2.22)
    # -- and an argument-position regex does not see them. check_paper is the
    # file whose whole job is to hold numbers against the data, so a literal in
    # it is a checked literal.
    cp = (R / "scripts/check_paper.py").read_text()
    got |= set(re.findall(r"-?\d+\.\d+", cp))
    return got


def main() -> int:
    have = known() | set(ALLOW) | section_numbers()
    meas = measured()
    bad = []
    # The supplement is checked against the measurements rather than against
    # macros: it computes nearly everything in f-strings, so what is left typed
    # is what deserves the question "is this still true anywhere?"
    import glob as _glob
    for f in sorted(_glob.glob(str(R / "scripts/supp/*.py"))):
        f = Path(f)
        for n, lines in sorted(prose_numbers(f).items()):
            if n in meas or n in have:
                continue
            bad.append((f.name, n, lines))
    for f in (R / "scripts/build_pdf.py",):
        for n, lines in sorted(prose_numbers(f).items()):
            if n in have:
                continue
            # a number whose macro prints it to fewer decimals still counts
            if any(k.startswith(n) or n.startswith(k) for k in have
                   if len(k) >= 3):
                continue
            bad.append((f.name, n, lines))
    for f, n, lines in bad:
        print(f"     {n} at line{'s' if len(lines) > 1 else ''} "
              f"{', '.join(str(x) for x in lines[:4])} is in no macro, no "
              f"claim and no allow-list")
    used_allow = sorted(
        n for n in prose_numbers(R / "scripts/build_pdf.py") if n in ALLOW
        and not ALLOW[n].startswith(("budget", "unit", "blend", "the headline")))
    for n in used_allow:
        print(f"     {n} allowed: {ALLOW[n]}")
    print(f"\n  {'PASS' if not bad else str(len(bad)) + ' unaccounted number(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
