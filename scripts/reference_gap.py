"""Which reference the quality is measured against, and what it is worth.

Section 5.1 makes a methodological point: a tiled decode compared against a
FULL-FRAME decode of the same model cancels the tiling penalty, and reports
the quality of a decoder nobody ships. The paper measures against the
released decoder instead, and quoted the difference between the two as
+0.035 dB at the deepest exit and +0.008 dB at the shallowest.

Those two numbers were typed into the prose from a run nobody kept, so they
could not move with the checkpoint. This measures them.

For a uniform-depth decode at exit j:

    d_release = 10 log10( mse(tiled_j) / mse(release) )
    tiling    = 10 log10( mse(tiled_j) / mse(full_frame_j) )

`tiling` is the quantity the prose quotes: what the cut costs at that depth,
which is exactly what cancels when both sides of the ratio are tiled. The
difference between the two columns at the deepest exit is the anchor drift,
and it is reported alongside because it is the check that the two agree
where they should.

    python scripts/reference_gap.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar
"""
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
import ctc_intra as C  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=16)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/reference_gap.json")
    a = ap.parse_args(argv)
    dev = a.device
    torch.cuda.set_device(dev)

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                      weights_only=False))
    K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[:a.max_seqs]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, exits {j} and {K - 1}", flush=True)

    exits = [j, K - 1]
    # Per rate as well as pooled: the tiling penalty at the deepest exit runs
    # from a third again to half again across the rate range, so a single
    # number for it has to say which rate it is.
    per = {qp_v: {e: {"tiled": 0.0, "full": 0.0} for e in exits}
           for qp_v in a.qps}
    per_ref = {qp_v: 0.0 for qp_v in a.qps}
    acc = {e: {"tiled": 0.0, "full": 0.0} for e in exits}
    acc_ref = 0.0
    with torch.no_grad():
        for qp_v in a.qps:
            for x in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                ph, pw = (-H) % P, (-W) % P
                xp = (F.pad(x, (0, pw, 0, ph), mode="replicate")
                      if (ph or pw) else x)
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                nt = ((H + ph) // P) * ((W + pw) // P)
                outs = net.dec.forward_all_exits(y, q, only=exits)
                for e in exits:
                    m = torch.full((nt,), e, dtype=torch.long, device=dev)
                    t_ = ((net.dec(y, q, exit_map=m) - xp) ** 2).mean().item()
                    f_ = ((outs[e] - xp) ** 2).mean().item()
                    acc[e]["tiled"] += t_; acc[e]["full"] += f_
                    per[qp_v][e]["tiled"] += t_; per[qp_v][e]["full"] += f_
                r_ = ((ref.dec.forward_full(y, q) - xp) ** 2).mean().item()
                acc_ref += r_; per_ref[qp_v] += r_

    def db(num, den):
        return 10 * torch.log10(torch.tensor(num / den)).item()

    rows = []
    for e in exits:  # shallowest first, deepest last
        d_rel = db(acc[e]["tiled"], acc_ref)
        d_own = db(acc[e]["tiled"], acc[e]["full"])
        rows.append({"exit": e, "db_vs_release": d_rel,
                     "tiling_db": d_own, "drift_db": d_rel - d_own})
        print(f"  exit {e}: vs release {d_rel:+.4f} dB, tiling alone "
              f"{d_own:+.4f}, difference {d_rel - d_own:+.4f}", flush=True)

    by_rate = []
    for qp_v in a.qps:
        row = {"qp": qp_v}
        for e in exits:
            row[f"tiling_db_exit{e}"] = db(per[qp_v][e]["tiled"],
                                           per[qp_v][e]["full"])
            row[f"db_vs_release_exit{e}"] = db(per[qp_v][e]["tiled"],
                                               per_ref[qp_v])
        by_rate.append(row)
        print(f"  q{qp_v:<3} tiling at exit {exits[0]} "
              f"{row[f'tiling_db_exit{exits[0]}']:+.4f}, at exit {exits[-1]} "
              f"{row[f'tiling_db_exit{exits[-1]}']:+.4f}", flush=True)

    _deep = [r[f"tiling_db_exit{exits[-1]}"] for r in by_rate]
    _shal = [r[f"tiling_db_exit{exits[0]}"] for r in by_rate]
    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "n_frames": len(frames), "qps": a.qps,
           "measured": "uniform-depth tiled decode scored against the released "
                       "decoder and against this model's own full-frame decode "
                       "at the same exit; the gap is the difference",
           "rows": rows,
           "by_rate": by_rate,
           "tiling_deep": rows[-1]["tiling_db"],
           "tiling_shallow": rows[0]["tiling_db"],
           "tiling_deep_lo": min(_deep), "tiling_deep_hi": max(_deep),
           "tiling_shallow_lo": min(_shal), "tiling_shallow_hi": max(_shal)}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
