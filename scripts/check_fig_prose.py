"""Every figure must be read, not merely named.

check_tex counts references, and a reference is not an explanation: "(Fig. 7)"
parked at the end of a clause tells the reader that a figure exists and nothing
about what to see in it. A figure earns its page if SOMETHING -- its caption or
the paragraph that cites it -- points at the picture: panels, axes, a line, a
colour, what to compare against what.

Both places count, because the paper legitimately uses both: some figures are
explained in a caption and pointed to in passing, others are named first and
read in the sentences that follow. What is not allowed is neither.

    python scripts/check_fig_prose.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
MIN_WORDS = 30
LIT = re.compile(r'r?f?"((?:[^"\\]|\\.)*)"')

READS = re.compile(
    r"\b(shows?|shown|showing|plots?|plotted|draws?|drawn|read|reads?|"
    r"compares?|against|panel|panels|left|right|top|bottom|axis|axes|"
    r"marked|marks|dashed|dotted|line|lines|point|points|colour|color|"
    r"traces?|curves?|bars?|markers?|horizontal|vertical|shaded|inset|"
    r"legend|column|row|band|region|each|per-|distribution|histogram)\b"
    # Nature-style panel labels are a reading instruction in themselves: "a,
    # the trained gate G" tells the reader which picture is being described.
    r"|(?:^|[.;] )[a-f], [a-z]", re.I)


def _call_text(s: str, start: int, limit: int = 3000) -> str:
    """The concatenated string literals of one call, stopping at its own ')'."""
    seg, depth = s[start:start + limit], 1
    end = len(seg)
    for i, ch in enumerate(seg):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    txt = "".join(x.group(1) for x in LIT.finditer(seg[:end]))
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = re.sub(r"\\\\?[A-Za-z]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def captions(path: Path) -> dict[str, str]:
    """name -> caption, for fig()/table() calls, keyed by the label they get."""
    s, out = path.read_text(), {}
    pat = r"\n\s*(?:story \+= )?(?:k\.)?(?:figure_wide|figure|fig)\(\s*r?f?\"([a-z0-9_]+)\.png\""
    for m in re.finditer(pat, s):
        out["fig:" + m.group(1)] = _call_text(s, m.end())
    for m in re.finditer(r"\n\s*(?:k\.)?tab(?:le)?\(", s):
        t = _call_text(s, m.end())
        out.setdefault("_tables", []).append(t)  # matched by number below
    return out


def blocks(path: Path) -> list[str]:
    s, out = path.read_text(), []
    for m in re.finditer(r"\n\s*(?:k\.)?(?:par|h2|note)\(", s):
        t = _call_text(s, m.end())
        if t:
            out.append(t)
    return out


def audit(builder: Path, label: str):
    caps = captions(builder)
    hits: dict[str, list[str]] = {}
    for b in blocks(builder):
        for m in re.finditer(r"\[\[(fig|tab):([a-z0-9_]+)\]\]", b):
            hits.setdefault(f"{m.group(1)}:{m.group(2)}", []).append(b)
    bad = []
    for name, cap in sorted(caps.items()):
        if name == "_tables" or name in hits:
            continue
        hits[name] = []                      # caption-only: the supplement
    for name, paras in sorted(hits.items()):
        cap = caps.get(name, "")
        pool = cap + " " + " ".join(paras)
        words = len(re.findall(r"[A-Za-z]{2,}", pool))
        # A table is read by its own columns; the reading test is for pictures.
        need_read = name.startswith("fig:")
        if words < MIN_WORDS or (need_read and not READS.search(pool)):
            why = f"{words} words" if words < MIN_WORDS else "nothing to read"
            bad.append((name, why, (cap or paras[0])[:120]))
    print(f"  {label}: {len(hits)} referenced, {len(bad)} thin")
    for n, why, ex in bad:
        print(f"     {n:26s} {why}\n        \"{ex}\"")
    return bad


def main():
    bad = audit(R / "scripts/build_pdf.py", "paper")
    for f in sorted((R / "scripts/supp").glob("[a-z]_*.py")):
        bad += audit(f, f"supp/{f.stem}")
    print(f"\n  {len(bad)} thin reference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
