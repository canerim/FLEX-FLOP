"""One line of truth about what is training right now.

Kept as a file rather than inlined in the monitor because both bugs it has
already had were in the *detection*, not the reporting, and a file can be tested
against reality before it is trusted.

Detection rules, each one earned
--------------------------------
1. Liveness comes from /proc, not from the log. A shelved run's last log line
   never changes, so a log-reading monitor keeps reporting its final numbers as
   if they were fresh. e1 stayed on the dashboard for hours after being stopped.

2. Match the script name without a leading slash, but require a python argv0.
   Anchoring on "/train_flexuf_image.py" looked like a tidy way to avoid matching
   shells that merely mention the script -- and silently stopped detecting every
   run launched with a RELATIVE path, which then showed up as DEAD while training
   perfectly well. Two false alarms are worse than one, so the filter is now the
   honest one: the process must actually be a python running that script.

3. The run's identity is the --save_dir argument, not any path that happens to
   start with runs/. "--pretrain .../runs/warmstart/ckpt_warmstart.pth.tar" also
   lives under runs/, and taking every matching argument invented a phantom run
   called "ckpt_warmstart.pth.tar".

4. Two runs per GPU is deliberate here, so it is not an alarm. The real failure
   -- the one that actually happened -- is a run alive when it should not be,
   holding a card its replacement was launched onto. That is alive - expected,
   and it has no false positives.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/home/can_karsal/FLEX-UF")
SHORT = {
    "baseline_singleexit": "BASE", "e1_j2_p128": "e1", "e2_j4_p128": "e2",
    "e3_j2_p64": "e3", "e4_ffn_adapter": "e4ffn",
    "wdec_j2_p128": "WD-j2/128", "wdec_j4_p256": "WD-j4/256",
    "wdec_j2_p128_arls": "ARLS-j2/128", "wdec_j2_p128_grid": "GRID-j2/128",
    "wdec_j2_p128_anchor10": "ANCHOR10",
    "wdec_j2_p256_grid": "GRID-j2/256",
    "wdec_j2_p256_distill": "DISTILL-j2/256",
    "heads_only_j2_p256": "HEADS-ONLY",
    "coupled_j2_p256": "COUPLED",
    "joint_j2_p256": "JOINT-ROUTER",
    "wdec_j2_p128_anchor10": "ANCHOR10",
    "wdec_j2_p256_dist_deepest": "D-deepest",
    "wdec_j2_p256_dist_plain": "D-plain1x1",
    "wdec_j2_p256_dist_fullrep": "D-fullrep",
    
}


def live_runs():
    """{run tag: pid} for every python actually running the trainer."""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = open(f"/proc/{p}/cmdline", "rb").read().decode(errors="ignore").split("\x00")
        except Exception:
            continue
        if not argv or "python" not in os.path.basename(argv[0]):
            continue
        if not any(a.endswith("train_flexuf_image.py") for a in argv):
            continue
        if "--save_dir" in argv:
            out[Path(argv[argv.index("--save_dir") + 1].rstrip("/")).name] = int(p)
    return out


def main():
    alive = live_runs()
    parts = []
    for tag in sorted(alive):
        f = ROOT / "runs" / tag / "train_log.jsonl"
        if not f.exists():
            continue
        try:
            rows = [json.loads(l) for l in f.open() if l.strip()]
        except Exception:
            continue
        if not rows:
            continue
        r = rows[-1]
        nm, deep = SHORT.get(tag, tag[:11]), r["psnr_per_exit"][-1]
        # Age of the last log line. Without it, a slow run that has not yet
        # reached its 200-step logging interval is indistinguishable from a
        # stalled one -- four runs showed identical step numbers on consecutive
        # ticks and looked frozen while every GPU sat at 100%.
        import time as _t
        age = int(_t.time() - f.stat().st_mtime)
        stale = f" STALE:{age}s" if age > 900 else ""
        if len(r["psnr_per_exit"]) == 1:
            parts.append(f"{nm} ep{r['epoch']} s{r['step']}({age}s) {deep:.1f}")
            continue
        # Fidelity to the frozen released decoder, on the SAME batch, as
        # 10*log10(1/anchor_mse). This is the number that had to be here.
        #
        # `deep` is the deepest exit's absolute PSNR and swings 7 dB with batch
        # content, so it cannot show drift. anchor_mse divides that content out.
        # FINE12 was found degrading -- 58.6 dB at step 1k falling to 54.7 by
        # 4k while every other run rose -- only because it was gone looking for
        # by hand. A trend in the one quantity every saving number is measured
        # against should not depend on someone thinking to check.
        import math as _m
        fid = None
        hist = [x.get("anchor_mse") for x in rows[-25:] if x.get("anchor_mse")]
        if hist:
            fid = 10 * _m.log10(1.0 / hist[-1])
        fnote = ""
        if len(hist) >= 8:
            # Compare the recent half against the earlier half rather than
            # consecutive points: single batches are noisy enough that any two
            # adjacent values can fall.
            h = len(hist) // 2
            old = sum(10 * _m.log10(1.0 / v) for v in hist[:h]) / h
            new = sum(10 * _m.log10(1.0 / v) for v in hist[h:]) / (len(hist) - h)
            if new < old - 1.0:
                fnote = f"!ANCHOR-FALLING({old:.1f}->{new:.1f})"
        # Spread alone is the wrong diagnostic: a small gap means either "nothing
        # to route" or "shallow exits caught up", and only the anchor separates
        # them. Both are flagged, neither assumed.
        note = ("!ANCHOR-WEAK" if deep < 20
                else "!EXITS-IDENTICAL" if r["spread_dB"] < 0.05 else "")
        fstr = f" fid{fid:.1f}" if fid is not None else ""
        parts.append(f"{nm} ep{r['epoch']} s{r['step']}({age}s)"
                     f"{fstr} spr{r['spread_dB']:+.1f}{note}{fnote}{stale}")

    exp_file = ROOT / "runs" / ".expected_live"
    expected = [l.strip() for l in exp_file.read_text().splitlines() if l.strip()] \
        if exp_file.exists() else []
    dead = [t for t in expected if t not in alive]
    if dead:
        parts.append("*** DEAD: " + ",".join(SHORT.get(d, d) for d in dead) + " ***")
    zombie = [t for t in alive if expected and t not in expected]
    if zombie:
        parts.append("*** ZOMBIE (olmemis, karti tutuyor): "
                     + ",".join(f"{SHORT.get(z, z)}:{alive[z]}" for z in zombie) + " ***")

    print(" | ".join(parts) if parts else "*** HICBIR EGITIM CALISMIYOR ***")


if __name__ == "__main__":
    main()
