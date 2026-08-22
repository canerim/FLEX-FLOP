"""The third axis: skip the stem where the picture is flat.

The width axis is answered and the answer is no (flexplus/FLEX-PLUS.pdf, 4.1).
What is left from the literature is spatial sparsity -- Dynamic Convolutions
(Verelst and Tuytelaars, arXiv:1912.03203) learn a mask and execute a
convolution only where it is set; focused convolutions (arXiv:2310.07782) do
the same for pretrained networks. Cost falls linearly with the active
fraction, which is weaker scaling than width's square, and it is the only one
of the three axes that changes no tensor shape and so adds no tile boundary.

The question this asks first, before any training: how much of the stem's work
is spent where nothing is happening? Rank positions by how much the stem
changes them -- the norm of its own residual -- and give the bottom fraction
the stem's input instead of its output. If the picture survives that, a mask
is worth learning. If it does not, this axis is answered too.

Like flexplus/stem_tolerance.py this is a QUALITY probe. The masked positions
are computed and then discarded, because what is asked is what they are worth,
not what skipping them saves. The cost of a real implementation is stated
arithmetically -- active fraction times the stem -- and never measured here.

    python flexplus/spatial_probe.py --device cuda:0
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
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402


def stem_masked(dec, y_hat, j, keep_frac):
    """The stem, with the least-changed (1 - keep_frac) of positions skipped.

    The ranking is the stem's own residual norm at each spatial position: how
    much the four blocks move that position. An oracle mask, deliberately --
    a learned mask can only do worse, so a failure here is conclusive and a
    success is an upper bound.
    """
    base = dec.upsample(y_hat)
    out = base
    for g in range(j):
        out = dec.groups[g](out)
    if keep_frac >= 1.0:
        return out, base
    delta = (out - base).pow(2).sum(1, keepdim=True)     # [1,1,H,W]
    flat = delta.flatten()
    k = max(1, int(round(flat.numel() * keep_frac)))
    thresh = torch.topk(flat, k, largest=True).values[-1]
    keep = (delta >= thresh).to(out.dtype)
    return base + keep * (out - base), base


def _decode(dec, feat, q, k_exit, cfg):
    for g in range(cfg.split_depth, k_exit + 1):
        feat = dec.groups[g](feat)
    return dec._apply_head(dec._at_exit(feat, k_exit), q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--keep", type=float, nargs="+",
                    default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--max_seqs", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(HERE / "results/spatial_probe.json"))
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    j = cfg.split_depth

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[:a.max_seqs]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, split depth {j}\n")
    print(f"  {'qp':>4}{'keep':>7}{'dB vs UF':>11}{'extra':>9}"
          f"{'stem cost':>11}{'ceiling':>10}")

    # The ceiling if a real implementation skipped the masked positions. The
    # stem is 0.2981 of a released decode and the floor with a full stem is
    # 0.6088 (flexplus/PLAN.md); nothing else in the floor moves.
    STEM, FLOOR = 0.2981, 0.6088
    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "split_depth": j,
           "n_frames": len(frames),
           "what": "oracle spatial mask on the always-on stem, quality only; "
                   "the cost column is arithmetic",
           "rows": []}
    with torch.no_grad():
        for qp_v in a.qps:
            base_db = None
            for keep in a.keep:
                tot = totref = 0.0
                for x in frames:
                    x = x.to(dev)
                    _, _, H, W = x.shape
                    P = cfg.rgb_patch
                    ph, pw = (-H) % P, (-W) % P
                    xp = (F.pad(x, (0, pw, 0, ph), mode="replicate")
                          if (ph or pw) else x)
                    qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                    y, q, _ = net._encode_to_latent(xp, qp)
                    feat, _ = stem_masked(net.dec, y, j, keep)
                    rec = _decode(net.dec, feat, q, j, cfg)
                    tot += ((rec - xp) ** 2).mean().item()
                    totref += ((ref.dec.forward_full(y, q) - xp) ** 2).mean().item()
                db = 10 * torch.log10(torch.tensor(tot / totref)).item()
                if base_db is None:
                    base_db = db
                cost = STEM * keep
                ceil = 100 * (1 - (FLOOR - STEM + cost))
                out["rows"].append({"qp": qp_v, "keep": keep, "db_vs_uf": db,
                                    "extra_db": db - base_db,
                                    "stem_cost_modelled": cost,
                                    "ceiling_modelled_pct": ceil})
                print(f"  {qp_v:>4}{keep:>7.2f}{db:>11.4f}{db - base_db:>+9.4f}"
                      f"{cost:>11.4f}{ceil:>9.2f}%")
            print()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
