"""What a trained narrow stem costs, and what ceiling it buys.

Two numbers per width, both measured rather than modelled:

  the price   dB below the released decoder with every tile on the shallowest
              rung, narrow stem against full stem, on the CTC intra frames
  the ceiling 1 - (MACs of that decode / MACs of a released decode), hooks on
              the executed pass, the same counter the paper reports

The second is the point of the exercise and the first is what it costs. A
width is worth having if its price fits inside the budget the paper reports at
-- 0.1 dB, or 0.2 dB for the target this branch was opened for.

    python flexplus/narrow_eval.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(ROOT / "scripts"))

import ctc_intra as C                                       # noqa: E402
from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.measure import MacMeter                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402
from narrow_stem import NarrowStem                          # noqa: E402


def _decode(dec, feat, q, k_exit, cfg):
    """Finish a decode from a stem feature, every tile at k_exit."""
    for g in range(cfg.split_depth, k_exit + 1):
        feat = dec.groups[g](feat)
    return dec._apply_head(dec._at_exit(feat, k_exit), q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="narrow stem .pth files; default is every one found")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(HERE / "results/narrow_eval.json"))
    a = ap.parse_args()
    dev = a.device

    stems = ([Path(s) for s in a.stems] if a.stems
             else sorted((HERE / "results").glob("narrow_stem_w*.pth")))
    if not stems:
        raise SystemExit("no trained narrow stem to evaluate")

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    j, K = cfg.split_depth, cfg.num_exits

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[:a.max_seqs]:
        x, _pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, split depth {j}\n")

    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "n_frames": len(frames), "rows": []}
    print(f"  {'width':>7}{'qp':>5}{'exit':>6}{'dB full':>10}"
          f"{'dB narrow':>11}{'extra':>9}{'ceiling':>10}")

    with torch.no_grad():
        for sp in stems:
            sd = torch.load(sp, map_location="cpu", weights_only=False)
            narrow = NarrowStem(sd["channels_full"], sd["width"]).to(dev)
            narrow.load_state_dict(sd["state_dict"])
            narrow.eval()
            for qp_v in a.qps:
                acc = {"full": 0.0, "narrow": 0.0, "ref": 0.0}
                macs = {"full": 0.0, "narrow": 0.0, "ref": 0.0}
                for x in frames:
                    x = x.to(dev)
                    _, _, H, W = x.shape
                    P = cfg.rgb_patch
                    ph, pw = (-H) % P, (-W) % P
                    xp = (F.pad(x, (0, pw, 0, ph), mode="replicate")
                          if (ph or pw) else x)
                    qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                    y, q, _ = net._encode_to_latent(xp, qp)

                    base = net.dec.upsample(y)
                    with MacMeter(net.dec) as m:
                        feat = base
                        for g in range(j):
                            feat = net.dec.groups[g](feat)
                        rec_f = _decode(net.dec, feat, q, j, cfg)
                    macs["full"] += m.total
                    acc["full"] += ((rec_f - xp) ** 2).mean().item()

                    with MacMeter(net.dec) as m1, MacMeter(narrow) as m2:
                        rec_n = _decode(net.dec, narrow(base), q, j, cfg)
                    # The upsample runs in both and is counted in neither of
                    # the two blocks above, so add it once explicitly.
                    with MacMeter(net.dec) as mu:
                        net.dec.upsample(y)
                    macs["narrow"] += m1.total + m2.total
                    macs["full"] += 0.0
                    acc["narrow"] += ((rec_n - xp) ** 2).mean().item()

                    with MacMeter(ref) as m:
                        rr = ref.dec.forward_full(y, q)
                    macs["ref"] += m.total
                    acc["ref"] += ((rr - xp) ** 2).mean().item()
                    macs["_upsample"] = macs.get("_upsample", 0.0) + mu.total

                db_f = 10 * torch.log10(
                    torch.tensor(acc["full"] / acc["ref"])).item()
                db_n = 10 * torch.log10(
                    torch.tensor(acc["narrow"] / acc["ref"])).item()
                ceil_f = 100 * (1 - macs["full"] / macs["ref"])
                ceil_n = 100 * (1 - (macs["narrow"] + macs["_upsample"])
                                / macs["ref"])
                out["rows"].append({
                    "width": sd["width"], "channels": sd["channels"],
                    "qp": qp_v, "exit": j,
                    "db_full_stem": db_f, "db_narrow_stem": db_n,
                    "extra_db": db_n - db_f,
                    "ceiling_full_pct": ceil_f, "ceiling_narrow_pct": ceil_n})
                print(f"  {sd['width']:>7.3g}{qp_v:>5}{j:>6}{db_f:>10.4f}"
                      f"{db_n:>11.4f}{db_n - db_f:>+9.4f}{ceil_n:>9.2f}%")
            print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
