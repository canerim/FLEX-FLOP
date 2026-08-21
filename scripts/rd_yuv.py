"""Rate-distortion per colour component, ours against the released decoder.

The paper quotes one PSNR, DCVC-UF's own 6:1:1 weighted YUV420 figure, and
sets the budget on it. That is the right number to report and the wrong number
to trust on its own: a weighted mean can be held at 0.1 dB while luma and
chroma move in opposite directions underneath it, and nothing in the budget
mechanism would notice. Chroma is a seventh of the weight and two thirds of
the planes.

So this measures Y, U and V separately, on the same frames, at the same
operating points, for the released decoder and for FLEX-UF at every budget.
The operating points are not re-derived: each row of the signalled sweep
records the Lagrange multiplier its bisection settled on, and the exit map is
a deterministic argmin given that multiplier, so the decode here is the decode
the sweep priced.

    python scripts/rd_yuv.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctc_intra as C                                       # noqa: E402
from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.cost import exit_costs                          # noqa: E402
from flexuf.eval import tiled_exit_mses                     # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402
from gpu import pick as _gpu                                # noqa: E402


def _psnr_components(x_hat, ref_planes):
    """(Y, U, V) PSNR in 4:2:0 on 0..255, the split of ctc_intra.psnr_611_420.

    Same downsample, same clamp, same reference planes, so 6Y+U+V over 8
    reproduces the number the rest of the paper quotes. Checked, not assumed:
    the caller asserts it against psnr_611_420 on the same tensors.
    """
    y_rec, uv_rec = C.yuv_444_to_420(x_hat + 0.5)
    y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()[0]
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    y, u, v = ref_planes
    return (C.calc_psnr(y, y_rec), C.calc_psnr(u, uv_rec[0]),
            C.calc_psnr(v, uv_rec[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--sweep", default="results/signalled_RECIPE512_ctc53.json")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/rd_yuv_PAPER.json")
    a = ap.parse_args()
    dev = a.device

    sweep = json.loads((ROOT / a.sweep).read_text())
    # {(qp, budget): lambda}, the multiplier that sweep's bisection settled on.
    lam_of = {(r["qp"], r["budget_db"]): r["lam"] for r in sweep["rows"]
              if r.get("budget_reachable") and r.get("lam") is not None}
    budgets = sorted({b for _, b in lam_of})
    if not budgets:
        raise SystemExit(f"{a.sweep} carries no reachable operating point")

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    cost = exit_costs(cfg, "head").to(dev)

    seqs, missing = C.discover([])
    frames_all, measured = [], []
    for sq in seqs:
        x, pl = C.read_frames(sq["path"], sq["w"], sq["h"], a.frames, 1)
        if x is None:
            continue
        for i in range(x.shape[0]):
            frames_all.append((x[i:i + 1], pl[i]))
        measured.append(sq["name"])
    print(f"  {len(frames_all)} CTC frames from {len(measured)} sequences "
          f"({len(missing)} not on disk), budgets {budgets}\n")
    print(f"  {'qp':>4}{'budget':>8}{'Y':>9}{'U':>9}{'V':>9}"
          f"{'6:1:1':>9}{'bpp':>9}")

    out = {"ckpt": a.ckpt, "sweep": a.sweep, "frames_per_seq": a.frames,
           "psnr_convention": "per component, 4:2:0 on 0..255, as "
                              "ctc_intra.psnr_611_420 before the 6:1:1 mean",
           "budgets": budgets, "rows": []}

    with torch.no_grad():
        for qp_v in a.qps:
            frames = frames_all
            acc = {}          # label -> [sumY, sumU, sumV, sum611, n]
            bpp_sum = n_f = 0.0
            for x, planes in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = (F.pad(x, (0, pw, 0, ph), mode="replicate")
                      if (ph or pw) else x)
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                bpp, _, _ = net._rate(aux, qp, H * W)
                bpp_sum += bpp.item()
                n_f += 1

                rec = ref.dec.forward_full(y, q)[:, :, :H, :W]
                yy, uu, vv = _psnr_components(rec, planes)
                w = (6 * yy + uu + vv) / 8.0
                # The split has to rebuild the number the paper quotes, or the
                # panels below are of some other metric.
                assert abs(w - C.psnr_611_420(rec, planes)) < 1e-6
                r = acc.setdefault("release", [0.0, 0.0, 0.0, 0.0, 0])
                r[0] += yy; r[1] += uu; r[2] += vv; r[3] += w; r[4] += 1

                M = tiled_exit_mses(net.dec, y, q, xp, cfg)
                for b in budgets:
                    lam = lam_of.get((qp_v, b))
                    if lam is None:
                        continue
                    k = (M + lam * cost[None, :]).argmin(1).clamp(
                        min=cfg.split_depth)
                    rec = net.dec(y, q, exit_map=k)[:, :, :H, :W]
                    yy, uu, vv = _psnr_components(rec, planes)
                    r = acc.setdefault(b, [0.0, 0.0, 0.0, 0.0, 0])
                    r[0] += yy; r[1] += uu; r[2] += vv
                    r[3] += (6 * yy + uu + vv) / 8.0; r[4] += 1

            bpp_mean = bpp_sum / n_f
            for label in ["release"] + budgets:
                if label not in acc:
                    continue
                sy, su, svv, sw, n = acc[label]
                row = {"qp": qp_v, "config": ("release" if label == "release"
                                              else "flexuf"),
                       "budget_db": None if label == "release" else label,
                       "psnr_y": sy / n, "psnr_u": su / n, "psnr_v": svv / n,
                       "psnr_611": sw / n, "bpp": bpp_mean, "n_frames": n}
                out["rows"].append(row)
                tag = "release" if label == "release" else f"{label:g} dB"
                print(f"  {qp_v:>4}{tag:>8}{row['psnr_y']:>9.3f}"
                      f"{row['psnr_u']:>9.3f}{row['psnr_v']:>9.3f}"
                      f"{row['psnr_611']:>9.3f}{bpp_mean:>9.4f}")

    out["n_sequences"] = len(measured)
    out["n_frames"] = len(frames_all)
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
