"""Where does the ladder run out of room, and what does that budget buy?

Every number so far is quoted at 0.1 dB, which is a choice rather than a
property of the system. The ladder has its own natural budget: the point at
which a rate saturates -- every cell already at the shallowest exit it may
take, so no further tolerance can be spent. Below that point the budget
binds; at it the ceiling is reached; above it nothing changes.

This finds that budget per rate, and reports the whole configuration against
the released DCVC-UF decoder at whichever budget is asked for: the release's
own bits and PSNR, what this decoder delivers, and what it saves.

CPU only, on the dumped tables.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices, allocate, per_frame_db  # noqa

RES = HERE / "results"
QPS = [0, 16, 32, 48, 63]
UF = Path.home() / "FLEX-UF"


def release_rows():
    d = json.loads((UF / "results/rd_absolute_PAPER.json").read_text())
    return {r["qp"]: r for r in d["rows"]}


class Sweep:
    def __init__(self, npz, j):
        C = Cost(json.loads((RES / "cost_constants.json").read_text()))
        te = load(npz)
        self.C = C
        self.M = te["M"].astype(np.float64)
        self.R = te["R"].astype(np.float64)
        self.K = int(te["K"][0]); self.j = j
        self.cell = int(te["cell"][0]); self.cf = self.cell // C.feature_stride
        self.ck = np.array([(k + 1) * C.blocks_per_exit for k in range(self.K)],
                           float)
        self.per = {}
        for q in QPS:
            fr = frame_slices(te["img"], te["qp"], q)
            sel = np.concatenate(fr)
            loc = {v: i for i, v in enumerate(sel)}
            frl = [np.array([loc[v] for v in f]) for f in fr]
            shapes = {}
            for rec in te["grid"]:
                g_, q_, nh_, nw_, _ = str(rec).split(",", 4)
                if int(q_) == q:
                    shapes[int(g_)] = (int(nh_), int(nw_))
            self.per[q] = (frl, sel, [shapes[i] for i in sorted(shapes)])

    def at(self, q, lam):
        frl, sel, shapes = self.per[q]
        M, R = self.M[sel], self.R[sel]
        k = allocate(M, self.ck, lam, self.j)
        db = per_frame_db(M, R, k, frl)
        maps = [k[f].reshape(*s) for f, s in zip(frl, shapes)]
        return db, self.C.saving(maps, self.cf), k

    def ceiling_db(self, q):
        """Distortion when every cell takes the shallowest exit it may."""
        return self.at(q, 1e9)[0]

    def ceiling_saving(self, q):
        return self.at(q, 1e9)[1]

    def operate(self, q, target):
        """Largest saving whose per-frame dB stays under target."""
        if self.ceiling_db(q) <= target:
            db, sv, k = self.at(q, 1e9)
            return db, sv, True
        lo, hi = 0.0, 1e-8
        while self.at(q, hi)[0] <= target and hi < 1e9:
            hi *= 4
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if self.at(q, mid)[0] <= target:
                lo = mid
            else:
                hi = mid
        db, sv, k = self.at(q, lo)
        return db, sv, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(RES / "cells_ctc64_e8.npz"))
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--out", default=str(RES / "budget_search.json"))
    a = ap.parse_args()

    s = Sweep(a.test, a.split)
    rel = release_rows()
    print(f"  hucre {s.cell}px, j={s.j}, tavan {s.ceiling_saving(0):.2f}%")
    print(f"\n  her oranin DOYMA butcesi (her hucre en sig cikista):")
    sat = {}
    for q in QPS:
        sat[q] = s.ceiling_db(q)
        print(f"   qp{q:>3}: {sat[q]:.4f} dB   tavan tasarrufu "
              f"{s.ceiling_saving(q):.2f}%")
    lo_sat = min(sat.values())
    print(f"\n  ilk doyma {lo_sat:.4f} dB'de (qp"
          f"{min(sat, key=sat.get)}), son doyma {max(sat.values()):.4f} dB'de")

    budgets = a.budgets or [0.10, round(lo_sat + 0.0005, 4), 0.20]
    out = {"cell_px": s.cell, "split_depth": s.j, "test": a.test,
           "saturation_db": {str(q): sat[q] for q in QPS}, "rows": []}
    for b in budgets:
        print(f"\n  === butce {b:.4f} dB ===")
        print(f"  {'qp':>4}{'bpp':>8}{'release':>10}{'bizim':>9}{'dPSNR':>9}"
              f"{'tasarruf':>10}{'doymus':>8}")
        for q in QPS:
            db, sv, saturated = s.operate(q, b)
            r = rel[q]
            ours = r["psnr_release"] - db
            print(f"  {q:>4}{r['bpp']:>8.4f}{r['psnr_release']:>10.4f}"
                  f"{ours:>9.4f}{-db:>9.4f}{sv:>9.2f}%{('  evet' if saturated else '  hayir'):>8}")
            out["rows"].append({"budget_db": b, "qp": q, "bpp": r["bpp"],
                                "psnr_release": r["psnr_release"],
                                "psnr_ours": ours, "db_below": db,
                                "saving_pct": sv, "saturated": bool(saturated)})
        m = np.mean([x["saving_pct"] for x in out["rows"]
                     if x["budget_db"] == b])
        print(f"  {'ort':>4}{'':>8}{'':>10}{'':>9}{'':>9}{m:>9.2f}%")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
