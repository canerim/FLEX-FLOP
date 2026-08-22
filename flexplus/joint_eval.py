"""Evaluate a jointly-trained pair: the narrow stem and the trunk that moved.

narrow_eval.py loads the paper's decoder and swaps in a narrow stem, which is
right when only the stem was trained. Joint training also moves the per-tile
trunk and the adapters, so the decoder to evaluate is the one that was
trained, not the pinned one. Loading the pinned decoder here would measure a
narrow stem against a trunk that never saw it and report the joint experiment
as a failure it is not.

    python flexplus/joint_eval.py --tag w0.5_joint --device cuda:0
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
sys.path.insert(0, str(HERE))

import ctc_intra as C                                       # noqa: E402
from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.measure import MacMeter                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402
from narrow_stem import NarrowStem                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    out = a.out or str(HERE / f"results/joint_eval_{a.tag}.json")

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    j = cfg.split_depth

    sd = torch.load(HERE / f"results/narrow_stem_{a.tag}.pth",
                    map_location="cpu", weights_only=False)
    narrow = NarrowStem(sd["channels_full"], sd["width"]).to(dev)
    narrow.load_state_dict(sd["state_dict"])
    narrow.eval()
    # The trunk as joint training left it.
    dec_p = HERE / f"results/joint_dec_{a.tag}.pth"
    moved = dec_p.exists()
    if moved:
        net.dec.load_state_dict(
            torch.load(dec_p, map_location="cpu", weights_only=False)["dec"])
        net.dec.eval()
    print(f"  {a.tag}: width {sd['width']}, "
          f"{'trunk from joint training' if moved else 'trunk from the pin'}")

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[:a.max_seqs]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    res = {"tag": a.tag, "width": sd["width"], "trunk_moved": moved,
           "n_frames": len(frames), "rows": []}
    print(f"  {'qp':>4}{'dB vs UF':>11}{'ceiling':>10}")
    with torch.no_grad():
        for qp_v in a.qps:
            tot = totref = 0.0
            m_n = m_r = 0.0
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
                with MacMeter(net.dec) as m1, MacMeter(narrow) as m2:
                    feat = narrow(base)
                    for g in range(j, j + 1):
                        feat = net.dec.groups[g](feat)
                    rec = net.dec._apply_head(net.dec._at_exit(feat, j), q)
                m_n += m1.total + m2.total
                tot += ((rec - xp) ** 2).mean().item()
                with MacMeter(ref) as m3:
                    rr = ref.dec.forward_full(y, q)
                m_r += m3.total
                totref += ((rr - xp) ** 2).mean().item()
            db = 10 * torch.log10(torch.tensor(tot / totref)).item()
            ceil = 100 * (1 - m_n / m_r)
            res["rows"].append({"qp": qp_v, "db_vs_uf": db,
                                "ceiling_pct": ceil})
            print(f"  {qp_v:>4}{db:>11.4f}{ceil:>9.2f}%")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
