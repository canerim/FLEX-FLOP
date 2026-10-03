"""The seam-repair audit, on CTC in YUV 6:1:1 -- the domain the paper's floor lives in.

The Kodak run said the trained grid repair costs rather than pays at the deepest
exit. Kodak is RGB and 768x512; the pinned numbers are CTC, YUV 6:1:1, mostly
1080p, and the frame border is a far smaller share of a 1080p frame. So the
finding has to be re-measured where it would be reported before it is believed.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path.home() / "DCVC"))

from ctc_intra import discover, read_frames, psnr_611_420        # noqa: E402
from flexuf.config import FlexUFConfig                           # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "RECIPE512" / "ckpt_PIN_e15.pth.tar"))
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--max_seq", type=int, default=12)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--out", default=str(ROOT / "results" / "seam_audit_ctc.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    repair = net.dec.seam_repair
    K, j = cfg.num_exits, cfg.split_depth

    seqs, _ = discover(a.classes)
    seqs = seqs[: a.max_seq]
    print(f"{len(seqs)} dizi, seam_repair={cfg.seam_repair}", flush=True)

    rows = []
    with torch.no_grad():
        for s in seqs:
            fr, planes = read_frames(s["path"], s["w"], s["h"], a.frames,
                                     max(1, min(a.stride, s["frames"] // max(a.frames, 1))))
            for fi in range(len(fr)):
                # read_frames returns the network input already shifted by -0.5
                # and the ORIGINAL 4:2:0 uint8 planes as numpy, which is what
                # psnr_611_420 compares against.
                x = fr[fi:fi + 1].to(dev)
                ref_planes = planes[fi]
                H, W = x.shape[-2:]
                al = cfg.rgb_patch
                xp = F.pad(x, (0, (-W) % al, 0, (-H) % al), mode="replicate")
                nt = (xp.shape[-2] // al) * (xp.shape[-1] // al)
                for qp_val in a.qps:
                    qp = torch.full((1,), qp_val, dtype=torch.int32, device=dev)
                    y, qd, _ = net._encode_to_latent(xp, qp)
                    p_ref = psnr_611_420(net.dec.forward_full(y, qd)[..., :H, :W], ref_planes)
                    deep = torch.full((nt,), K - 1, device=dev, dtype=torch.long)
                    torch.manual_seed(fi * 97 + qp_val)
                    mixed = torch.randint(j, K, (nt,), device=dev)
                    for tag, em in (("deepest", deep), ("routed", mixed)):
                        net.dec.seam_repair = repair
                        on = psnr_611_420(net.dec(y, qd, exit_map=em)[..., :H, :W], ref_planes)
                        net.dec.seam_repair = None
                        off = psnr_611_420(net.dec(y, qd, exit_map=em)[..., :H, :W], ref_planes)
                        net.dec.seam_repair = repair
                        rows.append({"seq": s["name"], "cls": s["cls"], "frame": fi,
                                     "qp": qp_val, "arm": tag, "w": s["w"], "h": s["h"],
                                     "gap_on_db": p_ref - on, "gap_off_db": p_ref - off,
                                     "repair_gain_db": on - off})
            print(f"  {s['name']} bitti", flush=True)

    for qp_val in a.qps:
        for tag in ("deepest", "routed"):
            g = [r for r in rows if r["qp"] == qp_val and r["arm"] == tag]
            if not g:
                continue
            f = lambda k: sum(r[k] for r in g) / len(g)
            print(f"  qp{qp_val:<3} {tag:<8}  ACIK {f('gap_on_db'):+.4f}  "
                  f"KAPALI {f('gap_off_db'):+.4f}  tamir {f('repair_gain_db'):+.4f} dB", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"yazildi {a.out}")


if __name__ == "__main__":
    main()
