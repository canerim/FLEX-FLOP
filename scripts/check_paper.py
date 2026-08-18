"""Does the paper reference anything that does not exist?

LaTeX fails loudly on a missing \\input and silently on a missing macro -- an
undefined \\MainLowRate typesets as nothing at all, so a number quietly vanishes
from a sentence that still reads like a sentence. This checks both, plus figures
and citation keys, without needing a LaTeX installation.
"""
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "paper"
SRC = [R / "main.tex", R / "supplementary.tex"]

BUILTIN = set("""dB note cvprfinalcopy item textbf emph cite ref label input
includegraphics caption begin end section subsection paragraph maketitle title
author documentclass usepackage bibliographystyle bibliography small large Large
centering toprule midrule bottomrule cmidrule multicolumn columnwidth textwidth
resizebox itemsep times arg min max log mathrm sum frac lVert rVert sg hat bmod
approx dagger etal IfFileExists newcommand textcolor vskip par null newpage quad
left right geq leq in mathbb theta lambda beta alpha Delta dots ProvidesPackage
RequirePackage""".split())

macros = set(re.findall(r"\\newcommand\{\\(\w+)\}",
                        (R / "tables" / "macros.tex").read_text()))
bibkeys = set(re.findall(r"@\w+\{([^,]+),", (R / "refs.bib").read_text()))

bad = []
for f in SRC:
    if not f.exists():
        continue
    t = f.read_text()
    for u in set(re.findall(r"\\([A-Z]\w+)", t)):
        if u not in macros and u not in BUILTIN and not u.startswith(("I", "P")):
            bad.append(f"{f.name}: undefined command \\{u}")
    for fig in set(re.findall(r"figures/([\w.]+)", t)):
        if not (R / "figures" / fig).exists():
            bad.append(f"{f.name}: missing figure {fig}")
    for tab in set(re.findall(r"\\input\{tables/(\w+)\}", t)):
        if not (R / "tables" / f"{tab}.tex").exists():
            bad.append(f"{f.name}: missing table tables/{tab}.tex")
    for key in set(k for grp in re.findall(r"\\cite\{([^}]+)\}", t)
                   for k in grp.split(",")):
        if key.strip() not in bibkeys:
            bad.append(f"{f.name}: undefined citation {key.strip()}")
    for lab in set(re.findall(r"\\ref\{([^}]+)\}", t)):
        if f"\\label{{{lab}}}" not in t:
            bad.append(f"{f.name}: dangling reference {lab}")

if bad:
    print("\n".join("  " + b for b in sorted(set(bad))))
    print(f"\n  {len(set(bad))} problem(s)")
    sys.exit(1)
print(f"  paper is self-consistent: {len(macros)} macros, "
      f"{len(bibkeys)} bib entries, all figures and tables present")
