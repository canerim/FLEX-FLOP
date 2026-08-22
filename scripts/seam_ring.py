"""What the deblocking pass earns, split by distance from the tile boundary.

Section 4.3 argues the pass does not earn its 0.95% of the decode. The
argument runs on a per-pixel decomposition -- a gain in the 0-4 px ring
around each tile boundary, a small loss everywhere else, and a ceiling on
what a perfect gate could earn -- and every number in it was typed into the
prose from a run nobody kept.

This measures the decomposition on a given checkpoint:

  * squared error against this model's own full-frame decode, per pixel,
    with the repair off and on, at uniform deepest depth;
  * split by distance to the nearest tile boundary;
  * the tiling penalty in decibels for each, and the penalty a PERFECT gate
    would leave -- the ring's error with the pass on, the interior's with it
    off, which is the best a position-indexed gate could do.

    python scripts/seam_ring.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=16)
    ap.add_argument("--ring", type=int, default=4,
                    help="width in pixels of the boundary ring")
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/seam_ring.json")
    a = ap.parse_args(argv)
    dev = a.device
    torch.cuda.set_device(dev)

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    K, P = cfg.num_exits, cfg.rgb_patch
    if net.dec.seam_repair is None:
        print("  this checkpoint has no seam repair")
        return 1

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[:a.max_seqs]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, ring {a.ring} px, tile {P} px", flush=True)

    keep = net.dec.seam_repair
    # Per rate as well as pooled. The seam is worst at the highest rate, so a
    # pooled verdict on whether the pass earns its cost can hide a rate where
    # it plainly does.
    per = {}
    tot = {k: 0.0 for k in ("ring_off", "ring_on", "in_off", "in_on",
                            "src_ring_off", "src_ring_on", "src_in_off",
                            "src_in_on", "ref", "n_ring", "n_in")}
    with torch.no_grad():
        for qp_v in a.qps:
            cur = {k: 0.0 for k in tot}
            for x in frames:
                x = x.to(dev)
                H0, W0 = x.shape[-2:]
                xp = F.pad(x, (0, (-W0) % P, 0, (-H0) % P), mode="replicate")
                H, W = xp.shape[-2:]
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                nt = (H // P) * (W // P)
                deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                net.dec.seam_repair = None
                off = net.dec(y, q, exit_map=deep)
                net.dec.seam_repair = keep
                on = net.dec(y, q, exit_map=deep)
                ref = net.dec.forward_full(y, q)

                yy = torch.arange(H, device=dev)[:, None].expand(H, W)
                xx = torch.arange(W, device=dev)[None, :].expand(H, W)
                dy = torch.minimum(yy % P, (P - 1) - (yy % P))
                dx = torch.minimum(xx % P, (P - 1) - (xx % P))
                # A pixel is in the ring if it is within `ring` of a boundary
                # in either direction, which is the band the gate acts on.
                ring = ((dy < a.ring) | (dx < a.ring))
                e_off = ((off - ref) ** 2).mean(1)[0]
                e_on = ((on - ref) ** 2).mean(1)[0]
                e_ref = ((ref - xp) ** 2).mean(1)[0]
                # The same split against the SOURCE rather than against the
                # full-frame decode. The prose's "gains 0.27% in the ring" is
                # a relative change, and which denominator it is relative to
                # changes it by two orders of magnitude, so both are recorded.
                s_off = ((off - xp) ** 2).mean(1)[0]
                s_on = ((on - xp) ** 2).mean(1)[0]
                tot["src_ring_off"] += s_off[ring].sum().item()
                tot["src_ring_on"] += s_on[ring].sum().item()
                tot["src_in_off"] += s_off[~ring].sum().item()
                tot["src_in_on"] += s_on[~ring].sum().item()
                tot["ring_off"] += e_off[ring].sum().item()
                tot["ring_on"] += e_on[ring].sum().item()
                tot["in_off"] += e_off[~ring].sum().item()
                tot["in_on"] += e_on[~ring].sum().item()
                tot["ref"] += e_ref.sum().item()
                tot["n_ring"] += int(ring.sum())
                tot["n_in"] += int((~ring).sum())
                for k in ("ring_off", "ring_on", "in_off", "in_on", "ref"):
                    pass
                cur["ring_off"] += e_off[ring].sum().item()
                cur["ring_on"] += e_on[ring].sum().item()
                cur["in_off"] += e_off[~ring].sum().item()
                cur["in_on"] += e_on[~ring].sum().item()
                cur["ref"] += e_ref.sum().item()
            per[qp_v] = cur

    n = tot["n_ring"] + tot["n_in"]
    ring_frac = tot["n_ring"] / n

    def db(err):
        # err is a summed squared error against the full-frame decode; the
        # penalty is that error carried on top of the reference's own.
        return 10 * float(np.log10((tot["ref"] + err) / tot["ref"]))

    off_db = db(tot["ring_off"] + tot["in_off"])
    on_db = db(tot["ring_on"] + tot["in_on"])
    # The best a gate indexed by position could do: take the pass where it
    # helps and leave it off where it hurts.
    best_db = db(min(tot["ring_off"], tot["ring_on"])
                 + min(tot["in_off"], tot["in_on"]))
    def _db(err, ref_):
        return 10 * float(np.log10((ref_ + err) / ref_))

    by_rate = []
    for qp_v in a.qps:
        c = per[qp_v]
        o = _db(c["ring_off"] + c["in_off"], c["ref"])
        n_ = _db(c["ring_on"] + c["in_on"], c["ref"])
        by_rate.append({"qp": qp_v, "penalty_off_db": o, "penalty_on_db": n_,
                        "recovered_db": o - n_})
        print(f"  q{qp_v:<3} penalty off {o:+.4f} on {n_:+.4f}  "
              f"recovered {o - n_:+.4f}", flush=True)

    out = {
        "by_rate": by_rate,
        "recovered_lo": min(r["recovered_db"] for r in by_rate),
        "recovered_hi": max(r["recovered_db"] for r in by_rate),
        "recovered_hi_qp": max(by_rate, key=lambda r: r["recovered_db"])["qp"],
        "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "n_frames": len(frames),
        "qps": a.qps, "ring_px": a.ring, "tile_px": P,
        "ring_fraction_of_pixels": ring_frac,
        "penalty_off_db": off_db, "penalty_on_db": on_db,
        "penalty_perfect_gate_db": best_db,
        "recovered_db": off_db - on_db,
        "perfect_gate_extra_db": on_db - best_db,
        "ring_error_change_pct": 100 * (tot["ring_on"] / tot["ring_off"] - 1),
        "interior_error_change_pct": 100 * (tot["in_on"] / tot["in_off"] - 1),
        "ring_error_change_vs_source_pct":
            100 * (tot["src_ring_on"] / tot["src_ring_off"] - 1),
        "interior_error_change_vs_source_pct":
            100 * (tot["src_in_on"] / tot["src_in_off"] - 1),
    }
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"  ring is {100 * ring_frac:.1f}% of pixels")
    print(f"  against the full-frame decode: ring {out['ring_error_change_pct']:+.2f}%, "
          f"interior {out['interior_error_change_pct']:+.2f}%")
    print(f"  against the source:            ring "
          f"{out['ring_error_change_vs_source_pct']:+.3f}%, interior "
          f"{out['interior_error_change_vs_source_pct']:+.3f}%")
    print(f"  tiling penalty: off {off_db:+.4f} dB, on {on_db:+.4f}, "
          f"perfect gate {best_db:+.4f}")
    print(f"  the pass recovers {out['recovered_db']:+.4f} dB; a perfect gate "
          f"would add {out['perfect_gate_extra_db']:+.4f}")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
