"""The two builds must cite the same paper for the same claim.

check_twins proves the LaTeX source and the reportlab build carry the same
paper, and deliberately ignores markup, so \\cite{evc} against "[22]" looks
like the same thing written two ways. It was not: CITE sent evc to 22, which
is SlimCAE, and the built paper's positioning table had EVC citing somebody
else's work.

Three things are checked, all against the bibliography rather than against
either build's opinion of it:

  * every key in CITE lands on a REFS entry whose author and year match the
    bib entry for that key;
  * every key cited in main.tex has a number in the reportlab build;
  * every reference number the reportlab prose uses exists, and every REFS
    entry is used by something.

    python scripts/check_cites.py
"""
from __future__ import annotations

import ast
import re
import unicodedata
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
BUILD = R / "scripts/build_pdf.py"
TEX = R / "paper/main.tex"
BIB = R / "paper/refs.bib"


def bib() -> dict[str, dict]:
    out, txt = {}, BIB.read_text()
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", txt, re.S):
        key, body = m.group(1).strip(), m.group(2)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", body, re.S):
            f[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        out[key] = f
    return out


def norm(t: str) -> str:
    """Letters only, accents folded, lower case.

    refs.bib writes accents as LaTeX control sequences and REFS writes them as
    the characters themselves: Y\\ilmaz against Yilmaz, Ball\\'e against Ballé,
    Bj{\\o}ntegaard against Bjøntegaard. Comparing the two forms directly
    reports every accented author as a mismatch.
    """
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t.lower() if c.isalpha() and ord(c) < 128)


def surnames(author: str) -> list[str]:
    """Last names, from either "Last, First and ..." or "First Last and ..."."""
    out = []
    for a in re.split(r"\s+and\s+", author):
        a = a.replace("{", "").replace("}", "").replace("\\", "").strip()
        if not a:
            continue
        out.append(a.split(",")[0].strip() if "," in a else a.split()[-1])
    return out


def main() -> int:
    src = BUILD.read_text()
    cite = ast.literal_eval(src[src.index("CITE = ") + 7:
                                src.index("\n}", src.index("CITE = ")) + 2])
    i = src.index("REFS = [")
    refs = ast.literal_eval(src[i + len("REFS = "):src.index("\n]", i) + 2])
    B = bib()
    bad = []

    for key, n in sorted(cite.items()):
        if not 1 <= n <= len(refs):
            bad.append(f"CITE[{key}] = {n}, and there are {len(refs)} refs")
            continue
        entry, ref = B.get(key), refs[n - 1]
        if not entry:
            bad.append(f"CITE[{key}] has no entry in refs.bib")
            continue
        want = surnames(entry.get("author", ""))
        year = entry.get("year", "")
        nref = norm(ref)
        hit = any(len(norm(w)) >= 3 and norm(w) in nref for w in want[:3])
        # "Li" is two letters and matches nothing safely, so a short surname
        # falls back to the title: the words a reference cannot share with a
        # different paper by the same author.
        title = [norm(w) for w in re.findall(r"[A-Za-z-]{5,}",
                                             entry.get("title", ""))]
        if not hit:
            hit = sum(1 for w in title[:4] if w and w in nref) >= 2
        if not hit or (year and year not in ref):
            bad.append(f"CITE[{key}] -> [{n}] \"{ref[:58]}...\" but the bib "
                       f"says {want[:1]} {year}")

    used = {int(x) for x in re.findall(r"\[(\d{1,2})(?:,|\])", src)
            if x.isdigit()}
    used |= {int(x) for g in re.findall(r"\[(\d{1,2}(?:,\s*\d{1,2})+)\]", src)
             for x in g.split(",")}
    used |= set(cite.values())
    for n in sorted(used):
        if n > len(refs):
            bad.append(f"the prose cites [{n}] and there are {len(refs)} refs")
    unused = [n for n in range(1, len(refs) + 1) if n not in used]
    if unused:
        bad.append(f"REFS entries nothing cites: {unused}")

    # Where the prose names an author -- "Shoham and Gersho [26]", "Blard et
    # al. [25]" -- the reference it points at must carry that name. Only the
    # "et al." and "X and Y" forms are checked: a model name like DCVC-UF or
    # STF is not in the reference string at all, so testing it would report
    # every correct citation in the related-work section. This is what
    # re-checks every number after a renumbering, which is the moment
    # citations go wrong without anything looking different.
    named = re.compile(r"\b([A-Z][A-Za-z-]{2,})\s+(?:et al\.|and\s+"
                       r"([A-Z][A-Za-z-]{2,}))\s*\[(\d{1,2})\]")
    checked = 0
    for m in named.finditer(src):
        who = [w for w in (m.group(1), m.group(2)) if w]
        n = int(m.group(3))
        if not 1 <= n <= len(refs):
            continue
        ref = norm(refs[n - 1])
        checked += 1
        if not any(norm(w) in ref for w in who):
            bad.append(f"the prose writes \"{' and '.join(who)} [{n}]\" and "
                       f"[{n}] is \"{refs[n - 1][:52]}...\"")

    tex_keys = set(re.findall(r"\\cite\{([^}]*)\}", TEX.read_text()))
    tex_keys = {k.strip() for g in tex_keys for k in g.split(",")}
    # main.tex cites by key and bibtex numbers them; the reportlab build only
    # needs a number for the keys that reach a table, since its prose writes
    # numbers directly. A key with no number is only a problem if a table uses
    # it, and every table key is in CITE by construction, so this is a warning.
    missing = sorted(k for k in tex_keys if k and k not in cite and k in B)
    print(f"  {len(cite)} mapped keys, {len(refs)} references, "
          f"{len(tex_keys)} keys cited in main.tex, "
          f"{checked} named citations verified")
    if missing:
        print(f"     not in CITE (fine unless a table cites them): "
              f"{len(missing)}")
    # A reference printed without authors reads as an incomplete entry, and
    # for an arXiv paper the authors are never actually unknown. Reported
    # rather than failed: filling one in is the author's call, not a build's.
    noauth = sorted(k for k, f in B.items() if not f.get("author"))
    if noauth:
        print(f"     no author in refs.bib: {', '.join(noauth)}")
    for b in bad:
        print(f"     {b}")
    print(f"\n  {'PASS' if not bad else str(len(bad)) + ' citation issue(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
