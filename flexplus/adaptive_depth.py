"""Per-position depth with a receptive field that is told the truth.

The paper's ladder cuts the frame into tiles and lets each tile leave the
trunk at its own depth. The cut is what costs: a 3x3 at a tile border reads
invented values, and that seam is 0.023 to 0.028 dB on the reported
checkpoint before any tile has exited. Section 4.4 shows the exact fix --
give each 3x3 its real neighbour -- is bit-identical at uniform depth and
destroys the allocation under routing, because a tile's neighbour is at a
different depth and the halo it reads is the wrong feature.

There is a version of the exchange that survives routing. A position that
stops early keeps being computed for as long as some deeper neighbour still
needs it, and only as far as that neighbour's receptive field reaches. Write
d(x) for the depth position x is allocated. Block b must be computed at x if
some y within b - c blocks of x is allocated depth >= b, which is the
max-plus dilation

    D(x) = max_y ( d(y) - dist(x, y) )

and the decode that runs block b exactly where D >= b is, by construction,
what every position would have got from a full-frame decode to its own
depth. No invented values, no seam, and the cost is the area of a band whose
width is the depth DIFFERENCE between neighbours rather than the whole tile.

The exit maps the router produces are spatially smooth -- that is why they
have contiguous regions rather than salt and pepper -- so the bands are thin.
On the five maps in results/ the band costs 1.8 to 9.6 per cent of a decode,
against a seam worth 0.023 to 0.089 dB depending on the split depth.

Nothing under ~/FLEX-UF is modified; the decoder is imported and read.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF))
sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402
import ctc_intra as C                                       # noqa: E402


def dilate_depth(blocks: torch.Tensor) -> torch.Tensor:
    """D(x) = max_y (b(y) - dist(x, y)), in BLOCKS, by repeated 3x3 max-pool.

    One block of the trunk is one 3x3, so its receptive field grows by one
    feature cell per block, and the arithmetic has to be in blocks: an exit is
    two of them. Counting in exits was the first version of this and it put
    the deepest uniform decode 0.58 dB off the released one, which is the
    check this function exists to pass.
    """
    out = blocks.float().clone()
    steps = int(out.max().item()) + 1
    for _ in range(steps):
        grown = F.max_pool2d(out[None, None], 3, stride=1, padding=1)[0, 0] - 1.0
        nxt = torch.maximum(out, grown)
        if torch.equal(nxt, out):
            break
        out = nxt
    return out


def decode_adaptive(net, y_hat, quant_step, depth_cells: torch.Tensor,
                    blocks_per_exit: int):
    """Run the trunk per position, to D, and read each position out at d.

    Returns the reconstruction and the fraction of the trunk's positions that
    had to be computed, which is what the saving is charged on.
    """
    dec = net.dec
    feat = dec.upsample(y_hat)
    H, W = feat.shape[-2:]
    e = depth_cells.to(feat.device).float()
    if e.shape != (H, W):
        e = F.interpolate(e[None, None], size=(H, W), mode="nearest")[0, 0]
    # exit e means groups 0..e have run, which is (e + 1) * blocks_per_exit
    # blocks. Block i is needed at x when some y within b(y) - i cells of x
    # needs at least i blocks, which is D >= i.
    b = (e + 1.0) * blocks_per_exit
    D = dilate_depth(b)

    n_pos = H * W
    computed = 0
    taken = {}
    K = dec.cfg.num_exits
    for g in range(K):
        first_block = g * blocks_per_exit + 1
        need = (D >= float(first_block) - 1e-6)
        if need.any():
            nxt = dec.groups[g](feat)
            m = need[None, None].to(feat.dtype)
            feat = feat * (1 - m) + nxt * m
            computed += int(need.sum().item()) * blocks_per_exit
        here = (e >= float(g) - 1e-6) & (e < float(g + 1) - 1e-6)
        if here.any():
            # Freeze this position's feature at ITS OWN exit. The dilation
            # keeps computing deeper here to serve a deeper neighbour, so by
            # the end of the loop feat holds a depth this position never asked
            # for -- and feeding that through the shallow adapter is what made
            # the exact decode read 0.015 dB worse than the tiled one.
            taken[g] = (here.clone(), dec._at_exit(feat, g))

    # Stitch the exit features first and run the head once, which is what the
    # deployed decoder does. Applying the head per exit and adding the pieces
    # moves the seam into the head: its 3x3 reads across an exit boundary and
    # sees a feature from a different depth, which cost 0.016 dB on the routed
    # map and made the exact decode look worse than the tiled one.
    stitched = None
    for g, (mask, e_feat) in taken.items():
        m = mask[None, None].to(e_feat.dtype)
        stitched = e_feat * m if stitched is None else stitched + e_feat * m
    out = dec._apply_head(stitched, quant_step)
    return out, computed / (n_pos * K * blocks_per_exit)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(HERE / "results/adaptive_depth.json"))
    a = ap.parse_args(argv)
    dev = a.device
    torch.cuda.set_device(dev)

    ck = torch.load(UF / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                      weights_only=False))
    P, K, j = cfg.rgb_patch, cfg.num_exits, cfg.split_depth
    bpe = 2

    seqs, _ = C.discover([])
    s_ = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
    x, _ = C.read_frames(s_["path"], s_["w"], s_["h"], 1, 1)
    x = x[0:1].to(dev)
    H0, W0 = x.shape[-2:]
    xp = F.pad(x, (0, (-W0) % P, 0, (-H0) % P), mode="replicate")
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)

    rows = []
    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        full = ref.dec.forward_full(y, q)
        base = ((full - xp) ** 2).mean().item()
        Hf, Wf = net.dec.upsample(y).shape[-2:]
        Fp = cfg.feature_patch
        nh, nw = Hf // Fp, Wf // Fp

        def db_of(rec):
            return 10 * torch.log10(
                torch.tensor(((rec - xp) ** 2).mean().item() / base)).item()

        for k in range(j, K):
            d = torch.full((Hf, Wf), float(k), device=dev)
            rec, frac = decode_adaptive(net, y, q, d, bpe)
            tiled = net.dec(y, q, exit_map=torch.full(
                (nh * nw,), k, dtype=torch.long, device=dev))
            rows.append({"kind": "uniform", "exit": k,
                         "db_per_position": db_of(rec),
                         "db_tiled": db_of(tiled),
                         "trunk_fraction": frac})
            print(f"  uniform exit {k}: per position {db_of(rec):+.4f} dB, "
                  f"tiled {db_of(tiled):+.4f}, trunk {frac:.3f}", flush=True)

        # The allocation the router actually makes on this frame, decoded both
        # ways. Same map, same weights; what differs is whether a 3x3 at a
        # border reads an invented value or a real one.
        import glob as _g
        for f in sorted(_g.glob(str(UF / "results/supp_exitmap_*.json"))):
            em = json.loads(Path(f).read_text())
            if a.seq.lower() not in em["seq"].lower() or em["qp"] != a.qp:
                continue
            m = torch.tensor(em["exit_map_row_major"], dtype=torch.long,
                             device=dev)
            if m.numel() != nh * nw:
                continue
            grid = m.reshape(nh, nw).float()
            per_cell = grid.repeat_interleave(Fp, 0).repeat_interleave(Fp, 1)
            rec, frac = decode_adaptive(net, y, q, per_cell, bpe)
            tiled = net.dec(y, q, exit_map=m)
            flat = float(sum((int(e) + 1) for e in m) / (m.numel() * K))
            rows.append({"kind": "routed", "map": Path(f).name,
                         "db_per_position": db_of(rec), "db_tiled": db_of(tiled),
                         "trunk_fraction": frac, "trunk_fraction_tiled": flat})
            print(f"\n  routed map from {Path(f).name}:")
            print(f"    tiled        {db_of(tiled):+.4f} dB, trunk {flat:.3f}")
            print(f"    per position {db_of(rec):+.4f} dB, trunk {frac:.3f}"
                  f"   (band costs {100 * (frac - flat):.1f}% of the trunk, "
                  f"buys {db_of(tiled) - db_of(rec):+.4f} dB)", flush=True)
    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "seq": s_["name"],
           "qp": a.qp, "K": K, "j": j, "blocks_per_exit": bpe,
           "what": "per-position depth with a max-plus dilated receptive field; "
                   "uniform depth is the exactness check",
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
