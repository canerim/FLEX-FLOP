"""How much the always-on stem can be degraded before the budget is gone.

The four trunk blocks below the split are 49% of the ceiling's floor and the
only part of it large enough to matter (flexplus/PLAN.md). They run full-frame,
before tiling, so the exit ladder cannot reach them: every tile pays for them
whatever depth it chooses.

This asks the one question that decides whether a second axis is worth
building. Degrade the stem -- run it at half resolution, or drop a fraction of
its channels -- and measure what that costs in dB at the shallowest exit, where
the ceiling lives. If a stem at a quarter of the cost spends more than the
whole 0.2 dB budget on its own, the axis is dead and no amount of engineering
will revive it. If it spends a fraction of it, the ceiling moves.

Two degradations, both training-free, so the answer arrives in minutes rather
than days:

  half   the stem runs on a 2x downsampled feature map and is bilinearly
         upsampled back -- RANet's argument (CVPR 2020) applied to a stem
         rather than to a classifier's input
  drop-f a fraction f of the stem's output channels is replaced by the
         stem's input on those channels, i.e. those channels skip the four
         blocks entirely -- the crudest possible slimming, and a lower bound
         on what a trained slimmable stem (ICLR 2019) would give

Both are deliberately untrained. A trained version can only do better, so a
negative result here is conclusive and a positive one is a floor.

    python flexplus/stem_tolerance.py --device cuda:0
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


def _stem(dec, y_hat, j, mode="full", frac=0.0):
    """The feature after the upsample and the j shared groups.

    `mode`:
      full   what the decoder does
      half   the shared groups run on a 2x downsampled map, then bilinear back
      drop   a fraction of channels bypass the shared groups entirely
    """
    feat = dec.upsample(y_hat)
    if mode == "full":
        for g in range(j):
            feat = dec.groups[g](feat)
        return feat
    if mode == "half":
        h, w = feat.shape[-2:]
        small = F.avg_pool2d(feat, 2)
        for g in range(j):
            small = dec.groups[g](small)
        return F.interpolate(small, size=(h, w), mode="bilinear",
                             align_corners=False)
    if mode == "drop":
        keep = feat.clone()
        out = feat
        for g in range(j):
            out = dec.groups[g](out)
        c = out.shape[1]
        n_drop = int(round(c * frac))
        if n_drop:
            # The dropped channels take the stem's input instead of its
            # output. Not a mask to zero: zeroing a channel is a much larger
            # perturbation than skipping four residual blocks on it, and would
            # answer a question nobody asked.
            out = out.clone()
            out[:, c - n_drop:] = keep[:, c - n_drop:]
        return out
    raise ValueError(mode)


def _decode_from_stem(dec, feat, quant_step, k_exit, cfg):
    """Finish the decode from a given stem feature, every tile at k_exit.

    Full-frame, not tiled: this probe is about the stem, and running the
    remaining groups full-frame keeps the tiling penalty out of the number so
    that what is measured is the stem's degradation and nothing else.
    """
    for g in range(cfg.split_depth, k_exit + 1):
        feat = dec.groups[g](feat)
    e = dec._at_exit(feat, k_exit)
    return dec._apply_head(e, quant_step)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(HERE / "results/stem_tolerance.json"))
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    j, K = cfg.split_depth, cfg.num_exits

    seqs, _ = C.discover([])
    seqs = seqs[:a.max_seqs]
    frames = []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, split depth {j}, {K} exits\n")

    variants = [("full", 0.0)] + [("half", 0.0)] + \
               [("drop", f) for f in (0.25, 0.5, 0.75)]
    out = {"ckpt": a.ckpt, "split_depth": j, "num_exits": K,
           "n_frames": len(frames), "qps": a.qps,
           "what": "dB below the released decoder at each exit, with the "
                   "always-on stem degraded", "rows": []}

    print(f"  {'qp':>4}{'exit':>6}{'stem':>10}"
          + "".join(f"{n:>10}" for n in ("dB vs UF", "extra dB")))
    with torch.no_grad():
        for qp_v in a.qps:
            base = {}
            for name, frac in variants:
                tag = name if name != "drop" else f"drop {frac:g}"
                for k_exit in (j, K - 1):
                    tot_mse = tot_ref = 0.0
                    for x in frames:
                        x = x.to(dev)
                        _, _, H, W = x.shape
                        P = cfg.rgb_patch
                        ph, pw = (-H) % P, (-W) % P
                        xp = (F.pad(x, (0, pw, 0, ph), mode="replicate")
                              if (ph or pw) else x)
                        qp = torch.full((1,), qp_v, dtype=torch.int32,
                                        device=dev)
                        y, q, _aux = net._encode_to_latent(xp, qp)
                        feat = _stem(net.dec, y, j, name, frac)
                        rec = _decode_from_stem(net.dec, feat, q, k_exit, cfg)
                        tot_mse += ((rec - xp) ** 2).mean().item()
                        rr = ref.dec.forward_full(y, q)
                        tot_ref += ((rr - xp) ** 2).mean().item()
                    db = 10 * torch.log10(
                        torch.tensor(tot_mse / tot_ref)).item()
                    key = (qp_v, k_exit)
                    if name == "full":
                        base[key] = db
                    extra = db - base.get(key, db)
                    out["rows"].append({
                        "qp": qp_v, "exit": k_exit, "stem": tag,
                        "db_vs_uf": db, "extra_db_vs_full_stem": extra})
                    print(f"  {qp_v:>4}{k_exit:>6}{tag:>10}"
                          f"{db:>10.4f}{extra:>10.4f}")
            print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
