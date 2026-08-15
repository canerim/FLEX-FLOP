"""Collect every frontier measurement into the one table the project is judged on.

Reads runs/*/frontier_*/frontier.tsv and prints, per experiment and checkpoint,
the compute saved against the PSNR lost -- plus the uniform-depth reference curve
each router has to beat.

Why the reference curve matters: if routing does not sit above "every tile at
exit k", the router is adding nothing and the honest conclusion is that a plain
shallower decoder would have done. That comparison is the difference between a
result and a number.

    python scripts/report.py [--runs runs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_frontier(tsv: Path):
    rows = []
    for line in tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({
            "beta": float(parts[0]),
            "saving": float(parts[1]),
            "db_loss": float(parts[2]),
            "share": parts[3],
        })
    return sorted(rows, key=lambda r: r["beta"])


def read_uniform(frontier_dir: Path):
    """The 'all tiles at exit k' curve, from any eval.json in this sweep."""
    for ev in sorted(frontier_dir.glob("beta_*/eval.json")):
        d = json.loads(ev.read_text())
        qp = next(iter(d["results"]))
        return d["results"][qp]["uniform"], d.get("control_max_diff")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path.home() / "FLEX-UF" / "runs"))
    a = ap.parse_args()
    runs = Path(a.runs)

    found = False
    for run_dir in sorted(runs.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_f = run_dir / "meta.json"
        cfg = json.loads(meta_f.read_text())["config"] if meta_f.exists() else {}

        for fdir in sorted(run_dir.glob("frontier_*")):
            tsv = fdir / "frontier.tsv"
            if not tsv.exists() or len(tsv.read_text().splitlines()) < 2:
                continue
            found = True
            print(f"\n{'=' * 72}")
            print(f"{run_dir.name}   {fdir.name.replace('frontier_', '')}")
            print(f"  j={cfg.get('split_depth')}  tile={cfg.get('latent_patch', 0) * 16}px  "
                  f"halo={cfg.get('latent_halo')}lat  adapter={cfg.get('adapter_kind')}")

            uni, ctrl = read_uniform(fdir)
            if ctrl is not None:
                flag = "OK" if ctrl == 0.0 else "!! NOT BIT-EXACT"
                print(f"  control: deepest exit vs stock UF  max|diff| = {ctrl}  {flag}")
            if uni:
                print(f"\n  reference — every tile at one depth:")
                print(f"    {'exit':>5} {'saving':>9} {'PSNR':>9}")
                base = uni[-1]["psnr"]
                for u in uni:
                    print(f"    {u['exit']:>5} {u['saving_pct']:>8.1f}% "
                          f"{u['psnr']:>8.2f}  ({u['psnr'] - base:+.2f} dB)")

            print(f"\n  routed frontier:")
            print(f"    {'beta':>6} {'saving':>9} {'dB lost':>9}   exit share")
            for r in read_frontier(tsv):
                print(f"    {r['beta']:>6.0f} {r['saving']:>8.2f}% "
                      f"{r['db_loss']:>+8.4f}   {r['share']}")

    if not found:
        print("no frontier measurements yet — they appear once the autopilot")
        print("finds a ckpt_epo*.pth.tar (first one lands at the end of epoch 0)")


if __name__ == "__main__":
    main()
