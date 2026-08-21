"""The supplement says every property that could fail silently has a test that
fails loudly. Run them and find out.

Section G.9 makes a claim about tests/ -- that they exist, that they assert at
zero tolerance, and how many properties test_equivalence.py holds. The count is
read from the file now, but the claim that they PASS was never checked by
anything, and a paper that says "asserted at zero tolerance" while the assertion
is red is worse than one that says nothing.

Needs a GPU, so it is not in the per-build check. Run it before a submission
and after any change to flexuf/.

    python scripts/check_tests.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent


def main() -> int:
    named = set()
    for f in sorted((R / "scripts/supp").glob("*.py")):
        # Not every test_*.py named in the supplement is ours: Section D
        # discusses ~/DCVC/test_video.py, which is Microsoft's. Only names
        # written as one of our tests count.
        txt = f.read_text()
        named |= set(re.findall(r"(?:tests/|<b>)(test_[a-z_]+\.py)", txt))
    missing = [n for n in sorted(named) if not (R / "tests" / n).exists()]
    for n in missing:
        print(f"     the supplement names tests/{n} and it does not exist")

    out = subprocess.run(
        [str(R / ".venv/bin/python"), "-m", "pytest", str(R / "tests"),
         "-q", "--no-header"],
        capture_output=True, text=True)
    tail = [l for l in out.stdout.splitlines() if l.strip()][-1:]
    line = tail[0] if tail else "no output"
    m = re.search(r"(\d+) passed", line)
    nfail = re.search(r"(\d+) failed", line)
    print(f"  {len(named)} test files named in the supplement, "
          f"{len(missing)} missing | pytest: {line.strip()}")
    if nfail or missing:
        for l in out.stdout.splitlines():
            if l.startswith("FAILED"):
                print(f"     {l}")
        print(f"\n  {(int(nfail.group(1)) if nfail else 0) + len(missing)} "
              f"problem(s)")
        return 1
    print(f"\n  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
