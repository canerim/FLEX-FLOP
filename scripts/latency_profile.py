"""Where does the tiled decode actually spend its time?

`scripts/latency.py` established the gap: the MAC model promises 27-42% and the
GPU delivers 13-34%. That is the what. This is the where, and without it any
attempt to close the gap is guesswork.

Three questions the aggregate timing cannot answer:

  1. How much is patchify/unpatchify -- pure data movement, no arithmetic, and
     entirely absent from the MAC model.
  2. How much is seam repair, which the cost model charges at 0.95% and which
     may cost much more in wall-clock because it is a small kernel on a large
     tensor.
  3. How much is lost inside the group loop itself, where each group launches on
     a shrinking set of tiles and the last groups run on a handful.

Measured by timing the same stages the decoder runs, in the same order, with
CUDA events around each -- not by torch.profiler, whose own overhead is large
relative to the small kernels this is trying to measure.

    python scripts/latency_profile.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.backbone.decoder import patchify, unpatchify  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/BEST/ckpt_eval.pth.tar")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--budget_db", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/latency_profile_BEST.json")
    a = ap.parse_args(argv)
    # torch.cuda.Event is created on the CURRENT device, not on the device the
    # tensors live on. Timing work on cuda:N with events on cuda:0 does not
    # error -- it returns numbers, and they are wrong. Every result this script
    # wrote before this line was added was measured with --device cuda:2 and
    # events on cuda:0.
    torch.cuda.set_device(a.device)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    dec, K, j = net.dec, cfg.num_exits, cfg.split_depth
    Fp = cfg.feature_patch

    curve = json.loads((ROOT / "results/curve_BEST.json").read_text())
    rows = [r for r in curve["rows"] if r["qp"] == a.qp and r.get("hist")]
    hist = min(rows, key=lambda r: abs(
        r.get("db_vs_uf_per_frame", 9) - a.budget_db))["hist"]

    W = (a.width + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    H = (a.height + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    n_tiles = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)
    props = torch.tensor(hist, dtype=torch.float) / sum(hist)
    counts = (props * n_tiles).round().long()
    counts[-1] += n_tiles - counts.sum()
    em = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                    for k, c in enumerate(counts)]).to(dev).clamp(min=j)

    stages, out = [], {}
    with torch.no_grad():
        x = torch.zeros(1, 3, H, W, device=dev)
        qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
        y, q, _ = net._encode_to_latent(x, qp)

        # 1. upsample + the shared stem: full-frame, unavoidable, and the same
        #    work the stock decoder does.
        def stem_fn():
            f = dec.upsample(y)
            for g in range(j):
                f = dec.groups[g](f)
            return f
        stages.append(("stem (upsample + groups 0..j-1)", timed(stem_fn,
                                                                a.warmup, a.iters)))
        feat = stem_fn()

        # 2. patchify: pure data movement, invisible to the MAC model.
        stages.append(("patchify", timed(lambda: patchify(feat, Fp),
                                         a.warmup, a.iters)))
        tiles, nh, nw = patchify(feat, Fp)

        # 3. the group loop, one entry per group, on the tile count that group
        #    actually sees. This is where the arithmetic intensity is lost.
        work = tiles
        active = torch.arange(n_tiles, device=dev)
        for g in range(j, K):
            n_here = int(active.numel())
            w_here = work

            def grp(gg=g, ww=w_here):
                return dec.groups[gg](ww)
            stages.append((f"group {g}  ({n_here} tiles)",
                           timed(grp, a.warmup, a.iters)))
            work = dec.groups[g](work)
            keep = em[active] > g
            active, work = active[keep], work[keep]
            if active.numel() == 0:
                break

        # 4. unpatchify + head + seam repair: full-frame again.
        canvas = torch.zeros_like(tiles)
        stages.append(("unpatchify", timed(
            lambda: unpatchify(canvas, nh, nw), a.warmup, a.iters)))
        merged = unpatchify(canvas, nh, nw)
        # Order matters and I had it backwards on the first pass: seam repair
        # runs on the STITCHED TRUNK FEATURE (384 channels), before the head,
        # not on the head's 3-channel output. Profiling it in the wrong place
        # would have priced a 384-channel 3x3 as a 3-channel one and reported
        # it as negligible.
        if getattr(dec, "seam_repair", None) is not None:
            stages.append(("seam repair (384ch, full-frame)", timed(
                lambda: dec.seam_repair(merged), a.warmup, a.iters)))
            merged = dec.seam_repair(merged)
        stages.append(("head", timed(
            lambda: dec._apply_head(merged, q), a.warmup, a.iters)))

    total = sum(t for _, t in stages)
    print(f"  qp {a.qp}, {a.budget_db:g} dB exit map, {n_tiles} tiles, "
          f"{H}x{W}\n")
    print(f"  {'stage':<34}{'ms':>9}{'share':>9}")
    for name, t in stages:
        print(f"  {name:<34}{t:>9.2f}{100*t/total:>8.1f}%")
    print(f"  {'-'*34}{total:>9.2f}{100:>8.1f}%")
    print("\n  Stages are timed in isolation, so they do not sum to the "
          "end-to-end\n  figure -- overlap and cache state differ. Use the "
          "SHARES, not the total.")
    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "qp": a.qp, "budget_db": a.budget_db,
         "n_tiles": n_tiles, "hist": hist,
         "device": torch.cuda.get_device_name(dev),
         "stages": [{"stage": n, "ms": t} for n, t in stages]}, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
