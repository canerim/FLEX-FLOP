"""A live macro and a typed number in one sentence is a defect waiting.

Twice in one day the same thing happened, both times because I made a number
live and left its neighbour typed. The head spread moved from 8.8 to 8.4 when
its convention was fixed, and the sentence resting on it still said "at some
rates" where the answer had become all five. \\EpochLatest and \\EpochGain
recomputed themselves when epoch 4 landed, and the list of values beside them
stopped at epoch 3 -- four numbers, a range of five epochs, and a gain taken
from a fifth the sentence never showed.

A number that updates itself beside one that does not is a worse arrangement
than two typed numbers, because it looks maintained. This lists the sentences
that are in that arrangement, so each can be justified or fixed.

Section, figure and table references are not numbers in this sense and are
skipped, as are the budget grid and plain units.

    python scripts/check_mixed.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
LIT = re.compile(r'r?f?"((?:[^"\\]|\\.)*)"')
MACRO = re.compile(r"\\\\?([A-Z][A-Za-z]+)")
NUM = re.compile(r"(?<![\d.\w-])(\d{1,4}\.\d{1,4})(?![\d.])")
SKIP = {"0.1", "0.2", "0.25", "0.3", "0.5", "0.05", "0.15", "0.75", "0.9",
        "1.0", "0.0", "2.0", "1.5", "0.4", "0.6", "0.8", "0.100"}
REFWORD = re.compile(r"(?:sections?|§|appendix|figs?\.|figures?|tables?|eq\.|"
                     r"equations?)\s+$", re.I)
# Reviewed and left typed, with the reason. Anything not on this list is new.
ALLOWED = {
    ("build_pdf.py", "0.036"): "the floor at q0, quoted beside the exchange's own drop",
    ("build_pdf.py", "0.003"): "the floor at q0 with the exchange on",
    ("build_pdf.py", "0.056"): "the floor at q63",
    ("build_pdf.py", "0.030"): "the floor at q63 with the exchange on",
    ("build_pdf.py", "0.075"): "map transfer, q63 map at q0",
    ("build_pdf.py", "19.8"): "map transfer, q63 map at q0",
    ("build_pdf.py", "4.4"): "a section reference split across literals",
    ("a_implementation.py", "10.7"): "AR(1) wall-clock share, stated once",
    ("d_full_results.py", "29.7"): "the pooled row, contrasted with the paper's",
    ("d_full_results.py", "5.126"): "a lambda, printed to identify a bisection",
    ("d_full_results.py", "5.150"): "a lambda, printed to identify a bisection",
    ("build_pdf.py", "10.7"): "AR(1) wall-clock share, stated once",
}


def prose(path: Path):
    src = path.read_text()
    out = []
    for m in re.finditer(r"\n\s*(?:k\.)?(?:par|h2|note|fig|tbl|figure)\(", src):
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
        out.append((src[:m.end()].count("\n") + 1,
                    re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t))))
    return out


def main() -> int:
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             (R / "paper/tables/macros.tex").read_text()))
    new = []
    for f in [R / "scripts/build_pdf.py"] + \
             [Path(x) for x in sorted(glob.glob(str(R / "scripts/supp/[a-z]_*.py")))]:
        for line, para in prose(f):
            for sent in re.split(r"(?<=[.!?])\s+", para):
                ms = [x for x in MACRO.findall(sent) if x in defined]
                if not ms:
                    continue
                for m in NUM.finditer(sent):
                    n = m.group(1)
                    if n in SKIP:
                        continue
                    if REFWORD.search(sent[max(0, m.start() - 12):m.start()]):
                        continue
                    if (f.name, n) in ALLOWED:
                        continue
                    new.append((f.name, line, n, ms[:2], sent[:88]))
    for f, l, n, ms, s in new:
        print(f"     {f}:{l} typed {n} beside {ms}\n        {s}")
    print(f"\n  {'PASS' if not new else str(len(new)) + ' mixed sentence(s)'}")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
