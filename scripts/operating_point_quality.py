"""What the routed decode delivers at the operating point, beyond the mean decibel.

Two questions a reviewer asks about a method whose entire budget is denominated
in PSNR, and which this project has never answered with a measurement.

1. Does a perceptual metric agree with the decibel?
   The budget is 0.1 dB of PSNR and the two decibel conventions in use here
   differ by roughly a third of it, so "0.1 dB" is doing a great deal of work.
   MS-SSIM is the cheapest independent check that the loss is as invisible as
   the decibel says. It is reported for the released decoder and for ours on the
   SAME latent, so the difference is the synthesis and nothing else.

2. Which tiles pay, and how much?
   The headline is a mean over tiles and a mean hides its own tail. This records
   the per-tile penalty of the real mixed decode, ranks the worst tiles, and
   names the sequence and the position each one came from, so a failure-case
   section can point at something specific rather than assert that the tail is
   small.

Both answers need the same decodes -- the per-exit table, the routed decode and
the released decode of one latent -- so they are measured in one pass rather
than in two, which halves the card time on a shared machine.

Method, and why
---------------
* The allocation is NOT re-derived here. Lambda per rate is read from the
  deployed measurement in results/signalled_*.json, whose two-level bisection
  already landed the whole test set on the budget, and the allocation rule
  reproduced is that file's own: argmin_k (mse_k + lambda C_k), clamped to the
  split depth. Re-bisecting would land on a slightly different operating point
  and the perceptual numbers would then not belong to the row they sit beside.
  The delivered decibel is recomputed all the same and printed against the
  stored one, so a mismatch is visible rather than silent.
* The penalty per tile comes from the REAL mixed decode, not from the per-exit
  table. The table decodes every tile at one common exit, so it cannot see what
  a tile suffers from a shallower neighbour across the seam, which is exactly
  the effect a worst-case hunt is looking for.
* MS-SSIM is computed on the luma plane in 4:2:0 at 0..255, the same domain
  scripts/../ctc_intra.py:psnr_611_420 computes PSNR in. Computing it in 4:4:4
  would compare against a chroma upsample of the source rather than the source.
  Luma only, because MS-SSIM is defined on a single channel and the weighted
  6:1:1 pooling used for PSNR has no counterpart in the MS-SSIM literature;
  stating that is better than inventing a pooling rule.
* MS-SSIM follows Wang, Simoncelli and Bovik (2003): five scales, an 11-tap
  Gaussian of sigma 1.5, valid-region filtering, the standard scale weights.
  The implementation is checked against its own definition at start-up
  (identical inputs must score exactly 1).

    python scripts/operating_point_quality.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar \
        --signalled results/signalled_RECIPE512_ctc53.json --budget 0.1 \
        --out results/supp_opquality_PAPER.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C  # noqa: E402
from src.utils.transforms import yuv_444_to_420  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.eval import per_tile_mse, tiled_exit_mses  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402

MS_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003, Table 1


def _gauss_window(size: int, sigma: float, device, dtype):
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _filt(x, w):
    """Separable Gaussian, valid region only (no padding).

    Padding the border invents pixels and then measures them; the reference
    implementation crops instead, and at 1080p the crop is 10 rows and columns.
    """
    C_ = x.shape[1]
    x = F.conv2d(x, w.view(1, 1, 1, -1).expand(C_, 1, 1, -1), groups=C_)
    return F.conv2d(x, w.view(1, 1, -1, 1).expand(C_, 1, -1, 1), groups=C_)


def _ssim_cs(x, y, w, c1, c2):
    """(SSIM, contrast-structure) means over the valid region."""
    mx, my = _filt(x, w), _filt(y, w)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx = _filt(x * x, w) - mx2
    sy = _filt(y * y, w) - my2
    sxy = _filt(x * y, w) - mxy
    cs = (2 * sxy + c2) / (sx + sy + c2)
    ssim = ((2 * mxy + c1) / (mx2 + my2 + c1)) * cs
    return ssim.mean(), cs.mean()


def ms_ssim(x, y, data_range: float = 255.0, win: int = 11, sigma: float = 1.5):
    """MS-SSIM of two [1, 1, H, W] tensors.

    Five scales need the image to survive four halvings and still exceed the
    window, so the smallest side must be at least 11 * 2**4 = 176. The CTC set's
    smallest class is 416x240, which clears it; anything smaller raises rather
    than silently dropping scales, because a 3-scale and a 5-scale number are
    not comparable and nothing downstream would know.
    """
    dev, dt = x.device, torch.float32
    x, y = x.to(dt), y.to(dt)
    n = len(MS_W)
    need = win * (2 ** (n - 1))
    if min(x.shape[-2:]) < need:
        raise ValueError(f"{tuple(x.shape[-2:])} too small for {n}-scale "
                         f"MS-SSIM (needs {need} px on the short side)")
    w = _gauss_window(win, sigma, dev, dt)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for i in range(n):
        s, cs = _ssim_cs(x, y, w, c1, c2)
        vals.append(cs.clamp_min(0.0) if i < n - 1 else s.clamp_min(0.0))
        if i < n - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    out = torch.ones((), device=dev, dtype=dt)
    for v, wt in zip(vals, MS_W):
        out = out * v ** wt
    return float(out)


def luma420(img: torch.Tensor) -> torch.Tensor:
    """The decoder's 4:4:4 tensor as the 4:2:0 luma plane on 0..255."""
    y_rec, _ = yuv_444_to_420(img + 0.5)
    return (y_rec * 255).clamp(0, 255)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--signalled", default="results/signalled_RECIPE512_ctc53.json",
                    help="the deployed run this operating point is read from; "
                         "its checkpoint must be the one measured here")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0,
                    help="stop after this many sequences; 0 means the whole "
                         "test set. For smoke tests only, and the count lands "
                         "in the result file so a truncated run cannot be "
                         "mistaken for a full one")
    ap.add_argument("--worst", type=int, default=30,
                    help="how many individual tiles to name")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = a.device

    # The implementation, against its own definition.
    probe = torch.rand(1, 1, 256, 256, device=dev) * 255
    self_test = ms_ssim(probe, probe)
    assert abs(self_test - 1.0) < 1e-5, f"MS-SSIM(x,x) = {self_test}"
    blur = F.avg_pool2d(F.pad(probe, (1, 1, 1, 1), mode="replicate"), 3, 1)
    self_test_blur = ms_ssim(probe, blur)
    print(f"  MS-SSIM self-check: identical {self_test:.6f}, "
          f"blurred {self_test_blur:.4f}")

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    # Both decoders must read the identical latent or the comparison is not the
    # one claimed. Same assertion every other measurement here makes.
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    worst_enc = max((sa[k] - sb[k]).abs().max().item() for k in sa)
    assert worst_enc == 0.0, f"encoders differ by {worst_enc}"

    sig = json.loads((ROOT / a.signalled).read_text())
    if sig.get("ckpt") != a.ckpt:
        raise SystemExit(f"{a.signalled} was measured on {sig.get('ckpt')}, not "
                         f"{a.ckpt}; the operating point would not be ours")
    lam_of = {r["qp"]: r["lam"] for r in sig["rows"]
              if abs((r.get("budget_db") or -1) - a.budget) < 1e-9
              and r.get("lam") is not None}
    missing_qp = [q for q in a.qps if q not in lam_of]
    if missing_qp:
        raise SystemExit(f"no {a.budget} dB row for qp {missing_qp} in {a.signalled}")
    db_stored = {r["qp"]: r["db_vs_uf"] for r in sig["rows"]
                 if abs((r.get("budget_db") or -1) - a.budget) < 1e-9}

    cost = exit_costs(cfg, "head").to(dev)
    K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch
    seqs, missing = C.discover([])
    if a.max_seqs:
        seqs = seqs[:a.max_seqs]
    print(f"  {len(seqs)} sequences on disk, {len(missing)} absent; "
          f"{P}px tiles, budget {a.budget:g} dB\n")
    print(f"  {'qp':>4}{'lam':>12}{'dB':>9}{'saved':>9}{'PSNR rel':>10}"
          f"{'PSNR ours':>11}{'MS-SSIM rel':>13}{'MS-SSIM ours':>14}")

    rows, per_seq, tiles_all = [], [], []
    with torch.no_grad():
        for qp_v in a.qps:
            lam = lam_of[qp_v]
            acc = {"bpp": 0.0, "psnr_rel": 0.0, "psnr_our": 0.0,
                   "ms_rel": 0.0, "ms_our": 0.0, "sv": 0.0, "db": 0.0, "n": 0}
            hist = torch.zeros(K, dtype=torch.long)
            for s in seqs:
                x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
                if x is None:
                    continue
                x = x[0:1].to(dev)
                planes = pl[0]
                _, _, H, W = x.shape
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                nh, nw = (H + ph) // P, (W + pw) // P
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                # Rate on the TRUE pixel count: the padding is our tiling's
                # artefact and the user never asked for those pixels.
                bpp = float(net._rate(aux, qp, H * W)[0].item())

                M = tiled_exit_mses(net.dec, y, q, xp, cfg)     # allocation only
                k = (M + lam * cost[None, :]).argmin(1).clamp(min=j)
                rel_full = ref.dec.forward_full(y, q)
                our_full = net.dec(y, q, exit_map=k)

                # Delivered decibel, recomputed rather than trusted.
                db = float(10 * torch.log10(((our_full - xp) ** 2).mean()
                                            / ((rel_full - xp) ** 2).mean()))
                rel_t = per_tile_mse(rel_full, xp, nh, nw, P)
                our_t = per_tile_mse(our_full, xp, nh, nw, P)
                pen = 10 * torch.log10(our_t / rel_t.clamp_min(1e-12))

                rel_c, our_c = rel_full[:, :, :H, :W], our_full[:, :, :H, :W]
                p_rel = C.psnr_611_420(rel_c, planes)
                p_our = C.psnr_611_420(our_c, planes)
                src_y = torch.from_numpy(planes[0]).to(dev).float()[None, None]
                m_rel = ms_ssim(luma420(rel_c), src_y)
                m_our = ms_ssim(luma420(our_c), src_y)
                sv = float(100 * (1 - cost[k].mean()))

                hist += torch.bincount(k.cpu(), minlength=K)
                pen_l = pen.tolist()
                k_l = k.tolist()
                for t, (pv, kv) in enumerate(zip(pen_l, k_l)):
                    # How much of this tile is replicated padding rather than
                    # picture. 1080 pads to 1280, so the bottom row of tiles on
                    # a 1080p frame is 200 of its 256 rows invented by
                    # F.pad(..., mode="replicate"), and a tile that is mostly
                    # invented smooth content looks cheap to route shallow.
                    # Recording it per tile is what lets the failure cases be
                    # attributed rather than described.
                    r_, c_ = t // nw, t % nw
                    pad_frac = (max(0, ph - (nh - 1 - r_) * P) / P
                                if r_ == nh - 1 else 0.0)
                    pad_frac += (max(0, pw - (nw - 1 - c_) * P) / P
                                 if c_ == nw - 1 else 0.0)
                    tiles_all.append({"qp": qp_v, "seq": s["name"],
                                      "cls": s["cls"], "tile": t,
                                      "row": r_, "col": c_,
                                      "nh": nh, "nw": nw, "exit": int(kv),
                                      "penalty_db": float(pv),
                                      "pad_fraction": float(min(1.0, pad_frac)),
                                      "mse_released": float(rel_t[t]),
                                      "mse_routed": float(our_t[t])})
                per_seq.append({"qp": qp_v, "seq": s["name"], "cls": s["cls"],
                                "res": f"{s['w']}x{s['h']}", "tiles": nh * nw,
                                "bpp": bpp, "db_vs_uf": db, "saving_pct": sv,
                                "psnr_released": p_rel, "psnr_routed": p_our,
                                "psnr_delta": p_our - p_rel,
                                "ms_ssim_released": m_rel,
                                "ms_ssim_routed": m_our,
                                "ms_ssim_delta": m_our - m_rel,
                                "worst_tile_penalty_db": float(pen.max()),
                                "mean_tile_penalty_db": float(pen.mean())})
                for key, v in (("bpp", bpp), ("psnr_rel", p_rel),
                               ("psnr_our", p_our), ("ms_rel", m_rel),
                               ("ms_our", m_our), ("sv", sv), ("db", db)):
                    acc[key] += v
                acc["n"] += 1
                del M, y, q, aux, rel_full, our_full, xp, x
            n = max(acc["n"], 1)
            row = {"qp": qp_v, "lam": lam, "n_sequences": acc["n"],
                   "budget_db": a.budget,
                   "db_vs_uf": acc["db"] / n,
                   "db_vs_uf_stored": db_stored.get(qp_v),
                   "saving_pct_vs_release": acc["sv"] / n,
                   "bpp": acc["bpp"] / n,
                   "psnr_released": acc["psnr_rel"] / n,
                   "psnr_routed": acc["psnr_our"] / n,
                   "psnr_delta": (acc["psnr_our"] - acc["psnr_rel"]) / n,
                   "ms_ssim_released": acc["ms_rel"] / n,
                   "ms_ssim_routed": acc["ms_our"] / n,
                   "ms_ssim_delta": (acc["ms_our"] - acc["ms_rel"]) / n,
                   "hist": hist.tolist()}
            # -10 log10(1 - MS-SSIM): the form a compression paper plots the
            # metric in, where differences a few thousandths below 1 are legible.
            row["ms_ssim_db_released"] = -10 * math.log10(
                max(1e-12, 1 - acc["ms_rel"] / n))
            row["ms_ssim_db_routed"] = -10 * math.log10(
                max(1e-12, 1 - acc["ms_our"] / n))
            rows.append(row)
            print(f"  {qp_v:>4}{lam:>12.3e}{row['db_vs_uf']:>9.4f}"
                  f"{row['saving_pct_vs_release']:>8.2f}%"
                  f"{row['psnr_released']:>10.4f}{row['psnr_routed']:>11.4f}"
                  f"{row['ms_ssim_released']:>13.6f}"
                  f"{row['ms_ssim_routed']:>14.6f}")

    # ---- the tail, which the mean hides --------------------------------
    tiles_all.sort(key=lambda t: -t["penalty_db"])
    tail = {}
    for qp_v in a.qps:
        ts = sorted((t["penalty_db"] for t in tiles_all if t["qp"] == qp_v))
        if not ts:
            continue
        nq = len(ts)
        tail[str(qp_v)] = {
            "n_tiles": nq,
            "mean_db": sum(ts) / nq,
            "median_db": ts[nq // 2],
            "p95_db": ts[min(nq - 1, int(0.95 * nq))],
            "p99_db": ts[min(nq - 1, int(0.99 * nq))],
            "max_db": ts[-1],
            "n_over_0_25_db": sum(1 for v in ts if v > 0.25),
            "n_over_0_5_db": sum(1 for v in ts if v > 0.5),
            "n_over_1_db": sum(1 for v in ts if v > 1.0)}
    # Where the tail comes from. Two candidate explanations for a tile that
    # gives up a lot -- it holds replicated padding, or it was sent to a
    # shallow exit -- and each is a column here rather than an anecdote drawn
    # from the ten worst rows.
    def stats(ts):
        ts = sorted(ts)
        if not ts:
            return None
        n = len(ts)
        return {"n": n, "mean_db": sum(ts) / n, "median_db": ts[n // 2],
                "p95_db": ts[min(n - 1, int(0.95 * n))], "max_db": ts[-1],
                "n_over_0_5_db": sum(1 for v in ts if v > 0.5)}

    by_padding, by_exit = {}, {}
    for qp_v in a.qps:
        ts = [t for t in tiles_all if t["qp"] == qp_v]
        by_padding[str(qp_v)] = {
            "no_padding": stats([t["penalty_db"] for t in ts
                                 if t["pad_fraction"] == 0.0]),
            "some_padding": stats([t["penalty_db"] for t in ts
                                   if t["pad_fraction"] > 0.0])}
        by_exit[str(qp_v)] = {
            str(k): stats([t["penalty_db"] for t in ts if t["exit"] == k])
            for k in range(K)}

    worst_seq = sorted(per_seq, key=lambda r: -r["worst_tile_penalty_db"])[:15]
    worst_ms = sorted(per_seq, key=lambda r: r["ms_ssim_delta"])[:15]

    print("\n  worst tiles, by the decibel they give up against the release:")
    for t in tiles_all[:10]:
        print(f"    {t['penalty_db']:>7.3f} dB  qp{t['qp']:<3} exit {t['exit']}  "
              f"tile {t['tile']:>3} (r{t['row']},c{t['col']})  {t['seq'][:38]}")
    print("\n  the tail, split by whether the tile holds replicated padding:")
    for qp_v in a.qps:
        d0 = by_padding[str(qp_v)]["no_padding"]
        d1 = by_padding[str(qp_v)]["some_padding"]
        if not (d0 and d1):
            continue
        print(f"    qp{qp_v:<3} picture only  n={d0['n']:>5} mean "
              f"{d0['mean_db']:.3f} p95 {d0['p95_db']:.3f} max {d0['max_db']:.3f}"
              f"   with padding  n={d1['n']:>5} mean {d1['mean_db']:.3f} "
              f"p95 {d1['p95_db']:.3f} max {d1['max_db']:.3f}")

    print("\n  sequences whose MS-SSIM falls furthest:")
    for r in worst_ms[:5]:
        print(f"    {r['ms_ssim_delta']:>+.6f}  qp{r['qp']:<3} "
              f"{r['saving_pct']:>5.1f}% saved  {r['seq'][:38]}")

    out = {"what": "perceptual metric beside PSNR, and the per-tile tail, at "
                   "one operating point",
           "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "ckpt_step": ck.get("step"),
           "operating_point_from": a.signalled, "budget_db": a.budget,
           "allocation": "argmin_k (mse_k + lam C_k), clamped to the split "
                         "depth; lam read from the deployed run",
           "psnr_convention": "(6Y+U+V)/8 in 4:2:0 on 0..255 (ctc_intra."
                              "psnr_611_420)",
           "ms_ssim_convention": "Wang et al. 2003, 5 scales, 11-tap Gaussian "
                                 "sigma 1.5, valid region, luma plane in 4:2:0 "
                                 "on 0..255",
           "ms_ssim_self_check": {"identical": self_test,
                                  "blurred_3x3_box": self_test_blur},
           "frames_per_seq": a.frames, "max_seqs": a.max_seqs, "n_sequences": len(per_seq) // max(len(a.qps), 1),
           "measured": sorted({r["seq"] for r in per_seq}),
           "not_measured": [m["name"] for m in missing],
           "rows": rows, "tail": tail,
           "tail_by_padding": by_padding, "tail_by_exit": by_exit,
           "worst_tiles": tiles_all[:a.worst],
           "worst_sequences_by_tile": worst_seq,
           "worst_sequences_by_ms_ssim": worst_ms,
           "per_sequence": per_seq}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
