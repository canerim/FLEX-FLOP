"""Intra RD on the CTC sequences, stock DCVC-UF vs the multi-exit routed decoder.

Why this exists separately from `rd_curve.py`
---------------------------------------------
`rd_curve.py` measures on held-out OpenImages crops, which is the right set for
"did training work" but the wrong set for "is this competitive". The codec
literature — and DCVC-UF's own README, whose headline is *bit saving over
VTM-17.0 on UVG* — reports on the common test conditions sequences. So the
comparison a reviewer will look for is this one.

Which sequences, and the honest limitation
------------------------------------------
`~/DCVC/test_cfg/all_yuv420.json` names UVG, MCL-JCV and HEVC classes B/C/D/E.
Of those, UVG is published freely by Ultra Video Group and HEVC class E's three
sequences are in xiph.org's derf collection. Classes B, C and D are behind
JVET participant credentials, so they are **not** measured here and no number
below should be read as covering them.

Padding, and why it is not the stock 16
---------------------------------------
`DMCI` pads to a multiple of 16. That is enough for the transform but not for
tiling: 1080 -> 1088 after 16-alignment, and 1088 / 128 = 8.5, so a 128px tile
grid does not fit and the last row of tiles would be a different shape. The
frame is therefore padded to a multiple of the **tile** size (1920x1152 for
128px tiles), which is what a real tiled deployment would do anyway. PSNR is
computed after cropping back to the true frame size, so the padding cannot
flatter the result — it only ever costs bits.

Both decoders read the same y_hat from the same encode, so bpp is identical by
construction and the vertical gap is the decoder alone. Same rule as `sweep_both`.

    python ctc_intra.py --ckpt runs/wdec_j2_p128_arls/ckpt.pth.tar \
        --router runs/.../router.pth.tar --qps 0 16 32 48 63 --frames 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.video_reader import YUV420Reader  # noqa: E402
from src.utils.transforms import ycbcr420_to_444_np, yuv_444_to_420  # noqa: E402
from src.utils.metrics import calc_psnr  # noqa: E402

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import saving  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.router.router import N_STEM_SIGNALS, ExitRouter, stem_signals  # noqa: E402

SEQ_ROOT = Path("/data10/shareddata/test_sequences/YUV")


def discover(classes) -> list[dict]:
    """Sequences actually present on disk, from DCVC's own test config.

    Reads the config rather than hard-coding names so the set stays whatever
    Microsoft says it is, and silently skips what has not been downloaded — with
    the caller printing what was skipped, because a quietly smaller test set is
    the easiest way to report a flattering average.
    """
    cfg = json.loads((DCVC_ROOT / "test_cfg" / "all_yuv420.json").read_text())
    out, missing = [], []
    for cls, spec in cfg["test_classes"].items():
        if classes and cls not in classes:
            continue
        base = SEQ_ROOT / spec.get("base_path", cls)
        for name, meta in spec.get("sequences", {}).items():
            p = base / name
            # Resolution lives in the filename: NAME_WxH_fps.yuv
            try:
                wh = [t for t in Path(name).stem.split("_") if "x" in t][0]
                w, h = (int(v) for v in wh.split("x"))
            except Exception:
                continue
            (out if p.exists() else missing).append(
                {"cls": cls, "name": name, "path": p, "w": w, "h": h,
                 "frames": meta.get("frames", 1)})
    return out, missing


def read_frames(path, w, h, n, stride):
    """`n` frames, `stride` apart.

    Returns the network input (4:4:4, /255, minus 0.5 — the exact chain in
    `test_video.py:113-121`) alongside the ORIGINAL 4:2:0 uint8 planes, because
    the reference for PSNR is the 4:2:0 source, not the 4:4:4 upsample of it.
    """
    r = YUV420Reader(str(path), w, h)
    xs, planes, idx = [], [], 0
    while len(xs) < n:
        y, uv = r.read_one_frame()
        if y is None:
            break
        if idx % stride == 0:
            xs.append(torch.from_numpy(ycbcr420_to_444_np(y, uv)).unsqueeze(0))
            planes.append((y[0], uv[0], uv[1]))
        idx += 1
    if not xs:
        return None, None
    x = torch.cat(xs, dim=0).float() / 255.0 - 0.5
    return x, planes


def psnr_611_420(x_hat, ref_planes) -> float:
    """DCVC-UF's own metric: (6*Y + U + V)/8, computed in 4:2:0 on 0..255.

    `test_video.py:31-44` shifts the reconstruction back by +0.5, downsamples it
    to 4:2:0, scales to 0..255 and compares against the source's uint8 planes.
    Measuring in 4:4:4 instead would compare against a chroma upsample of the
    source rather than the source, and would report a different (generally
    higher) chroma PSNR — not comparable to any published DCVC-UF number.
    """
    y_rec, uv_rec = yuv_444_to_420(x_hat + 0.5)
    y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()[0]
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    y, u, v = ref_planes
    return (6 * calc_psnr(y, y_rec) + calc_psnr(u, uv_rec[0])
            + calc_psnr(v, uv_rec[1])) / 8.0


@torch.no_grad()
def eval_frame(net, router, cfg, x, ref_planes, qp_val, device):
    """One frame at one QP through both decoders. Returns (bpp, dense, routed, saved)."""
    _, _, H, W = x.shape
    # Pad to a whole number of tiles (see module docstring). The RGB-domain tile
    # side is the alignment that matters; 16-alignment would leave a half tile.
    align = cfg.rgb_patch
    ph, pw = (-H) % align, (-W) % align
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x

    qp = torch.full((1,), qp_val, dtype=torch.int32, device=device)
    y_hat, q_dec, aux = net._encode_to_latent(xp, qp)
    # Rate is charged on the TRUE pixel count, not the padded one: the padding is
    # an artefact of our tiling, and billing the codec for pixels the user never
    # asked for would understate bpp.
    bpp, _, _ = net._rate(aux, qp, H * W)

    dense = net.dec.forward_full(y_hat, q_dec)[:, :, :H, :W]
    d_psnr = psnr_611_420(dense, ref_planes)

    r_psnr, sv = None, 0.0
    if router is not None:
        stem = net.dec.upsample(y_hat)
        for g in range(cfg.split_depth):
            stem = net.dec.groups[g](stem)
        sc = aux["scales_hat"]
        if sc.shape[1] != y_hat.shape[1]:
            sc = sc[:, : y_hat.shape[1]]
        sig = stem_signals(stem, y_hat, sc, cfg)
        em = router.assign(sig, qp.long().repeat_interleave(sig.shape[0]))
        routed = net.dec(y_hat, q_dec, exit_map=em)[:, :, :H, :W]
        r_psnr = psnr_611_420(routed, ref_planes)
        sv = saving(em, cfg, "head")
    return bpp.item(), d_psnr, r_psnr, 100 * sv


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--router", default=None)
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=4, help="intra frames per sequence")
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--out", default="results/ctc_intra.json")
    a = ap.parse_args(argv)

    seqs, missing = discover(a.classes)
    if not seqs:
        raise SystemExit(f"no sequences under {SEQ_ROOT}")
    print(f"{len(seqs)} sequences present; {len(missing)} not downloaded")
    if missing:
        by = {}
        for m in missing:
            by[m["cls"]] = by.get(m["cls"], 0) + 1
        print("  NOT MEASURED: " + ", ".join(f"{k} ({v})" for k, v in sorted(by.items())))

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)

    router = None
    if a.router:
        rk = torch.load(a.router, map_location="cpu", weights_only=False)
        router = ExitRouter(cfg.num_exits, n_signals=N_STEM_SIGNALS,
                            min_exit=cfg.split_depth).to(a.device).eval()
        router.load_state_dict(rk["router"])

    rows = []
    for s in seqs:
        fr, planes = read_frames(s["path"], s["w"], s["h"],
                         a.frames, max(1, min(a.stride, s["frames"] // max(a.frames, 1))))
        if fr is None:
            print(f"  {s['name']}: unreadable, skipped"); continue
        for qp_val in a.qps:
            acc = [0.0, 0.0, 0.0, 0.0]
            for i in range(fr.shape[0]):
                x = fr[i:i + 1].to(a.device)
                b, d, r, sv = eval_frame(net, router, cfg, x, planes[i], qp_val, a.device)
                acc[0] += b; acc[1] += d; acc[3] += sv
                acc[2] += r if r is not None else 0.0
            n = fr.shape[0]
            row = {"cls": s["cls"], "seq": s["name"], "qp": qp_val, "n": n,
                   "bpp": acc[0] / n, "dense": acc[1] / n,
                   "routed": (acc[2] / n) if router else None,
                   "saved_pct": acc[3] / n}
            rows.append(row)
            msg = f"  {s['cls']:>7} {Path(s['name']).stem[:26]:<26} qp{qp_val:>2}  bpp {row['bpp']:.4f}  dense {row['dense']:.2f}"
            if router:
                msg += f"  routed {row['routed']:.2f} ({row['routed']-row['dense']:+.3f} dB, {row['saved_pct']:.1f}%)"
            print(msg, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "router": a.router, "config": cfg.__dict__,
         "measured": [s["name"] for s in seqs],
         "not_measured": [m["name"] for m in missing], "rows": rows}, indent=2))
    print(f"\nwrote {a.out}")

    print(f"\n  {'class':>7} {'qp':>3} {'bpp':>8} {'dense':>7}" +
          (f" {'routed':>7} {'dPSNR':>8} {'saved':>7}" if router else ""))
    for cls in sorted({r["cls"] for r in rows}):
        for qp_val in a.qps:
            g = [r for r in rows if r["cls"] == cls and r["qp"] == qp_val]
            if not g:
                continue
            m = lambda k: sum(r[k] for r in g) / len(g)  # noqa: E731
            line = f"  {cls:>7} {qp_val:>3} {m('bpp'):>8.4f} {m('dense'):>7.2f}"
            if router:
                line += (f" {m('routed'):>7.2f} {m('routed')-m('dense'):>+8.3f} "
                         f"{m('saved_pct'):>6.1f}%")
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
