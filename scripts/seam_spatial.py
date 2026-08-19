"""Where grid seam repair acts, and what its gate actually learned.

The module is gated by position within a tile, so it is *supposed* to correct
the boundary ring and switch itself off in the interior. Whether it does is a
different question from whether the frame average improves, and the frame
average cannot answer it: a module that fixes the seam and damages the interior
averages out to roughly nothing, which is roughly what the frame numbers said.

Two measurements, both off one pinned checkpoint so the figure they feed cannot
disagree with itself:

  gate    the trained gate read straight out of the checkpoint, summarised as a
          profile against distance from the tile edge, beside the exp(-d/tau)
          it was initialised with. The initialisation is the module's own prior
          about where it should act, so it is the right thing to compare against.

  effect  the per-pixel error split by distance from the nearest tile boundary,
          repair OFF against repair ON with the same exit map. The map is fixed
          first by bisecting lambda to the 0.1 dB operating point, so both sides
          decode the same tiles at the same depths and the only difference is
          the module itself.

This exists because the numbers behind the grid-seam-repair figure were read off
`runs/BEST/ckpt_eval.pth.tar`, which a watcher overwrites every epoch, so they
could not be reproduced afterwards. Everything here names a pinned checkpoint
and records its epoch.

    ./.venv/bin/python scripts/seam_spatial.py --device cuda:2
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path.home() / "DCVC"))

# Distance from the nearest tile boundary, in RGB pixels. The first band is the
# ring the module is meant to repair; the rest is the interior it is meant to
# leave alone. 128 is half a 256 px tile, so the four bands tile the frame.
BANDS = [(0, 4), (4, 16), (16, 64), (64, 128)]


def gate_profile(ck) -> dict:
    """The trained gate as a function of distance from the tile edge.

    The gate is one scalar per feature cell of a tile, shared over all 384
    channels, so it is small enough to report in full; what the figure needs is
    the profile, because the claim under test is about distance and nothing else.
    """
    from flexuf.config import FlexUFConfig
    from flexuf.backbone.decoder import GridSeamRepair

    cfg = FlexUFConfig(**ck["config"])
    sd = ck.get("model", ck.get("state_dict", ck))
    G = sd["dec.seam_repair.gate"].detach().float().squeeze().cpu().numpy()
    P = cfg.feature_patch
    idx = np.arange(P)
    d1 = np.minimum(idx, P - 1 - idx).astype(float)
    D = np.minimum(d1[:, None], d1[None, :])

    # tau is GridSeamRepair's own default and is not stored in the config, so it
    # is read off the class rather than typed in again here.
    tau = float(inspect.signature(GridSeamRepair.__init__).parameters["tau"].default)

    px = cfg.rgb_patch // P          # RGB pixels per feature cell
    out = {"feature_patch": P, "tile_px": cfg.rgb_patch, "px_per_cell": px,
           "tau_init": tau, "distance_cells": [], "distance_px": [],
           "n_cells": [], "trained_mean": [], "trained_min": [],
           "trained_max": [], "init": []}
    for k in range(P // 2):
        m = G[D == k]
        out["distance_cells"].append(k)
        # A cell at feature distance k covers RGB pixels k*px .. k*px+px-1 from
        # the edge, so its centre is the honest x for a pixel axis.
        out["distance_px"].append(k * px + px / 2)
        out["n_cells"].append(int(m.size))
        out["trained_mean"].append(float(m.mean()))
        out["trained_min"].append(float(m.min()))
        out["trained_max"].append(float(m.max()))
        out["init"].append(float(np.exp(-k / tau)))
    out["ring_mean"] = out["trained_mean"][0]
    out["ring_max"] = out["trained_max"][0]
    out["interior_mean"] = float(G[D >= 2].mean())
    out["interior_min"] = float(G[D >= 2].min())
    out["interior_max"] = float(G[D >= 2].max())
    return out


def measure_effect(ck, ckpt_path, n_seq, qp, budget_db, device) -> dict:
    """Error by distance from the nearest tile boundary, repair OFF against ON."""
    import ctc_intra as C
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.reference import reference_for

    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(device).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(device).eval()
    ref_path = reference_for(cfg, None)
    load_flexuf_state(ref, torch.load(ref_path, map_location="cpu",
                                      weights_only=False))
    cost = exit_costs(cfg, "head").to(device)
    j, P = cfg.split_depth, cfg.rgb_patch

    seqs, _ = C.discover([])
    seqs = seqs[:n_seq]
    acc = {b: {True: 0.0, False: 0.0} for b in BANDS}
    npx = {b: 0 for b in BANDS}
    used = []
    with torch.no_grad():
        for s in seqs:
            x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
            if x is None:
                continue
            used.append(s["name"])
            x = x[0:1].to(device)
            _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            Hp, Wp = xp.shape[-2:]
            qpt = torch.full((1,), qp, dtype=torch.int32, device=device)
            y, q, _ = net._encode_to_latent(xp, qpt)
            full = ref.dec.forward_full(y, q)
            nh, nw = Hp // P, Wp // P

            def tile_mse(im):
                return (((im - xp) ** 2).mean(1).view(1, nh, P, nw, P)
                        .permute(0, 1, 3, 2, 4).reshape(nh * nw, P * P).mean(1))

            M = torch.stack([tile_mse(o) for o in net.dec.forward_all_exits(y, q)], 1)
            Rm = tile_mse(full).mean()

            # Fix the operating point before comparing, so repair OFF and repair
            # ON decode exactly the same tiles at exactly the same depths and the
            # module is the only difference between the two frames.
            def at(lam):
                return (M + lam * cost[None, :]).argmin(1).clamp(min=j)

            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                db = (10 * torch.log10(
                    M.gather(1, at(mid)[:, None]).squeeze(1).mean() / Rm)).item()
                if db <= budget_db:
                    lo = mid
                else:
                    hi = mid
            em = at(lo)

            r = torch.arange(Hp, device=device) % P
            c = torch.arange(Wp, device=device) % P
            dr = torch.minimum(r, P - 1 - r)
            dc = torch.minimum(c, P - 1 - c)
            dist = torch.minimum(dr[:, None], dc[None, :])

            keep = net.dec.seam_repair
            err = {}
            for on in (False, True):
                net.dec.seam_repair = keep if on else None
                err[on] = ((net.dec(y, q, exit_map=em) - xp) ** 2).mean(1)[0]
            net.dec.seam_repair = keep

            for b in BANDS:
                m = (dist >= b[0]) & (dist < b[1])
                npx[b] += int(m.sum())
                for on in (False, True):
                    acc[b][on] += float(err[on][m].sum())

    total = sum(npx.values())
    off = [acc[b][False] / npx[b] for b in BANDS]
    on = [acc[b][True] / npx[b] for b in BANDS]
    share = [100.0 * npx[b] / total for b in BANDS]
    change = [100.0 * (o - f) / f for f, o in zip(off, on)]
    # The frame average is the share-weighted mean over bands, not the mean of
    # the per-band percentages, because the bands do not carry equal error.
    off_mean = sum(s * f for s, f in zip(share, off)) / 100.0
    on_mean = sum(s * o for s, o in zip(share, on)) / 100.0
    return {"checkpoint": str(ckpt_path), "epoch": int(ck.get("epoch", -1)),
            "reference": str(ref_path), "qp": qp, "budget_db": budget_db,
            "sequences": used, "n_sequences": len(used),
            "bands_px": [list(b) for b in BANDS], "share_pct": share,
            "mse_off": off, "mse_on": on, "change_pct": change,
            "frame_mse_off": off_mean, "frame_mse_on": on_mean,
            "frame_change_pct": 100.0 * (on_mean - off_mean) / off_mean,
            "frame_change_db": 10.0 * float(np.log10(on_mean / off_mean))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    # A pinned checkpoint by default. `ckpt_eval.pth.tar` is rewritten every
    # epoch by the watcher, so a number read off it cannot be reproduced later.
    ap.add_argument("--ckpt", default="runs/BEST/ckpt_epo0.pth.tar")
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--out", default="results/seam_spatial.json")
    a = ap.parse_args(argv)

    path = R / a.ckpt
    ck = torch.load(path, map_location="cpu", weights_only=False)
    res = {"checkpoint": str(path), "epoch": int(ck.get("epoch", -1)),
           "gate": gate_profile(ck)}
    res["gate"]["checkpoint"] = str(path)
    res["gate"]["epoch"] = int(ck.get("epoch", -1))
    if not a.gate_only:
        res["effect"] = measure_effect(ck, path, a.seqs, a.qp, a.budget, a.device)

    (R / a.out).write_text(json.dumps(res, indent=1))

    g = res["gate"]
    print(f"  gate   {path.name} epoch {res['epoch']}: ring {g['ring_mean']:.4f} "
          f"(max {g['ring_max']:.4f}), interior {g['interior_mean']:.4f}")
    if "effect" in res:
        e = res["effect"]
        print(f"  effect qp{e['qp']}, {e['budget_db']} dB, {e['n_sequences']} sequences")
        print(f"  {'band (px)':<12}{'share':>8}{'OFF':>12}{'ON':>12}{'change':>10}")
        for b, s, f, n, c in zip(e["bands_px"], e["share_pct"], e["mse_off"],
                                 e["mse_on"], e["change_pct"]):
            print(f"  {f'{b[0]}-{b[1]}':<12}{s:>7.1f}%{f:>12.6f}{n:>12.6f}{c:>+9.2f}%")
        print(f"  frame average {e['frame_change_pct']:+.4f}% "
              f"({e['frame_change_db']:+.5f} dB)")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
