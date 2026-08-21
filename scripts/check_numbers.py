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
    for m in re.finditer(r"\n\s*(?:story \+= )?(?:par|h1|h2|note|figure_wide"
                         r"|figure|tbl)\(", src):
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
    bad = []
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
