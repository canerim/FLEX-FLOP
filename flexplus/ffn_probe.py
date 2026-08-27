"""Is the FFN already a mixture of four experts?

Two thirds of this decoder is pointwise FFN: 5C^2 of a block's 7C^2 + 9C, and
the trunk is 89.4% of the decode. Whatever is done about compute has to be
done there, and the mixture-of-experts literature says the way to do it is to
route among parts of the FFN rather than run all of it.

Usually that means carving experts out of a trained FFN by clustering
co-activating neurons -- MoEfication, CMoE. Here the carving may already be
done. The block's FFN is

    conv1: C -> 4C,  WSiLU,  chunk-add,  conv2: C -> C

and the chunk-add is not a reshape but a SUM over four strided groups:

    out[j] = sum_{r=0..3} WSiLU( h[4j + r] )

so every output slot is an additive mixture of four sub-units, and the four
groups partition the hidden layer exactly. Dropping group r means not
computing rows 4j+r of conv1 -- one quarter of the expand -- and the FFN falls
from 5C^2 to (k+1)C^2 for k groups kept.

This measures whether the four groups are interchangeable or specialised: how
much of the sum's energy each carries, how correlated they are, and what the
block's output error is when one is dropped. If they are near-equal and highly
correlated, routing between them buys little and the honest answer is no. If
they are skewed or complementary, there is something to route.

Read-only: one checkpoint, a few frames, no weight written. CPU by default.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
sys.path.insert(0, str(UF)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(UF / "scripts"))
import ctc_intra as C                                            # noqa: E402
from flexuf.config import FlexUFConfig                           # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state          # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PIN_e9.pth.tar"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RES / "ffn_probe.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    dec = net.dec

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[: a.frames]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append((s["name"], x[0:1]))
    print(f"  {len(frames)} kare, epoch {ck.get('epoch')}, {dev}", flush=True)

    K = cfg.num_exits
    blocks = [g for grp in dec.groups for g in grp]
    print(f"  {len(blocks)} DepthConvBlock", flush=True)

    stats = {}
    with torch.no_grad():
        for name, x in frames:
            x = x.to(dev)
            _, _, H, W = x.shape
            P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            for q in a.qps:
                qp = torch.full((1,), q, dtype=torch.int32, device=dev)
                y, qs, _ = net._encode_to_latent(xp, qp)
                feat = dec.upsample(y)
                for bi, blk in enumerate(blocks):
                    if blk.adaptor is not None:
                        feat = blk.adaptor(feat)
                    inner = blk.dc(feat) + feat
                    h = blk.ffn[0](inner)                     # [1, 4C, h, w]
                    g = blk.ffn[1].wsilu(h)                   # WSiLU, still 4C
                    grp = [g[:, r::4] for r in range(4)]      # four C-channel groups
                    s = sum(grp)
                    out_full = blk.ffn[2](s) + inner
                    rec = stats.setdefault(bi, {
                        "energy": np.zeros(4), "absmean": np.zeros(4),
                        "corr": np.zeros((4, 4)), "drop_rel": np.zeros(4),
                        "sum_absmean": 0.0, "n": 0})
                    for r in range(4):
                        rec["energy"][r] += float((grp[r] ** 2).mean())
                        rec["absmean"][r] += float(grp[r].abs().mean())
                        # what the block loses if this group is never computed
                        out_r = blk.ffn[2](s - grp[r]) + inner
                        rec["drop_rel"][r] += float(
                            ((out_r - out_full) ** 2).mean()
                            / (out_full ** 2).mean())
                    v = torch.stack([grp[r].reshape(-1) for r in range(4)])
                    v = v - v.mean(1, keepdim=True)
                    cm = (v @ v.t()) / (v.norm(dim=1)[:, None]
                                        * v.norm(dim=1)[None, :] + 1e-12)
                    rec["corr"] += cm.cpu().numpy()
                    rec["sum_absmean"] += float(s.abs().mean())
                    rec["n"] += 1
                    feat = blk.ffn[2](s) + inner
            print(f"   {name[:34]:<36} bitti", flush=True)

    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "frames": [n for n, _ in frames], "qps": a.qps,
           "what": "the four strided groups of the chunk-add, per trunk block",
           "blocks": []}
    print(f"\n{'blok':>5}{'enerji payi (4 grup)':>34}{'grup-arasi korelasyon':>24}"
          f"{'bir grup dusurulunce bagil hata':>32}")
    for bi in sorted(stats):
        r = stats[bi]; n = r["n"]
        e = r["energy"] / e_sum if (e_sum := r["energy"].sum()) else r["energy"]
        cm = r["corr"] / n
        off = cm[np.triu_indices(4, 1)]
        dr = r["drop_rel"] / n
        out["blocks"].append({
            "block": bi, "energy_share": e.tolist(),
            "absmean": (r["absmean"] / n).tolist(),
            "corr_offdiag_mean": float(off.mean()),
            "corr_offdiag_min": float(off.min()),
            "drop_rel_err": dr.tolist()})
        print(f"{bi:>5}   " + " ".join(f"{100*x:5.1f}%" for x in e)
              + f"{off.mean():>18.3f}   " + " ".join(f"{x:.4f}" for x in dr))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
