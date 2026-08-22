"""Three numbers the coupling section typed by hand, measured.

Section 4.4 says the halo exchange is exact: at uniform depth a tiled decode
with the exchange is bit-identical to a full-frame one, once the comparison is
not confounded by the seam-repair filter, which runs on the stitched frame and
has no full-frame counterpart. The paper gave 1.13e-2 with the filter on,
exactly 0 with it off, and 6.06e-2 for replicate padding. Those three numbers
were typed into the prose from a run nobody kept, so they could not move with
the checkpoint and did not: on the pinned weights the first of them is wrong by
a factor of four.

This measures all three the same way, over the same frames, and writes them
where the prose can read them.

    python scripts/halo_exactness.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--max_seqs", type=int, default=16)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/halo_exactness.json")
    a = ap.parse_args(argv)
    dev = a.device
    torch.cuda.set_device(dev)

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg0 = FlexUFConfig(**ck["config"])
    K, P = cfg0.num_exits, cfg0.rgb_patch

    def build(coupled, seam):
        c = FlexUFConfig(**{**cfg0.__dict__, "tile_coupling": coupled})
        m = FlexUFIntra(c).to(dev).eval()
        load_flexuf_state(m, ck)
        if not seam:
            # The filter runs on the stitched frame and has no full-frame
            # counterpart, so leaving it in measures the filter, not the seam.
            m.dec.seam_repair = None
        return m

    nets = {"halo exchange, filter on": build(True, True),
            "halo exchange, filter off": build(True, False),
            "replicate padding, filter off": build(False, False)}
    full = build(False, True)

    seqs, _ = C.discover([])
    seqs = seqs[:a.max_seqs]
    frames = []
    for s in seqs:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, {len(a.qps)} rates", flush=True)

    worst = {k: 0.0 for k in nets}
    per_qp = []
    with torch.no_grad():
        for qp_v in a.qps:
            here = {k: 0.0 for k in nets}
            for x in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = full._encode_to_latent(xp, qp)
                nt = ((H + ph) // P) * ((W + pw) // P)
                deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)
                # The full-frame decode the tiled ones are compared against
                # never sees a tile boundary, so the filter is not in it.
                ref = full.dec.forward_full(y, q)
                for name, net in nets.items():
                    d = (ref - net.dec(y, q, exit_map=deep)).abs().max().item()
                    here[name] = max(here[name], d)
            for k, v in here.items():
                worst[k] = max(worst[k], v)
            per_qp.append({"qp": qp_v, **here})
            print(f"  qp {qp_v:>2}  " + "   ".join(
                f"{k}: {v:.3e}" for k, v in here.items()), flush=True)

    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "n_frames": len(frames), "qps": a.qps,
           "measured": "max|full-frame decode - tiled decode| at uniform "
                       "depth, over frames and rates",
           "worst": worst, "rows": per_qp}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    for k, v in worst.items():
        print(f"    {k:<32} {v:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
