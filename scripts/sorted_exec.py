"""Can the per-group bookkeeping be removed by sorting the tiles once?

The routed decode realises less wall-clock than its arithmetic predicts, and
none of the difference appears in the MAC model. Part of it is structural: at every group boundary the
decoder computes a boolean mask, gathers the surviving tiles and scatters the
finished ones, and boolean indexing has to know how many elements survive, which
forces a device-to-host synchronisation. Four groups, four syncs.

The fix, if it works, is to sort once:

  * order the tiles by exit depth, DESCENDING, so the tiles that go deepest come
    first;
  * then "the tiles still active at group g" is a contiguous PREFIX, and each
    group is a slice rather than a gather;
  * the slice boundaries are a cumulative histogram of the exit map, computed
    once, so there is one sync for the whole loop instead of one per group;
  * finished tiles are written back through the inverse permutation in a single
    scatter at the end.

The arithmetic is identical -- the same tiles run through the same groups. Only
the bookkeeping changes. This benchmark measures both against each other on the
same tensors, so a difference is attributable.

Standalone on purpose: `flexuf/backbone/decoder.py` is imported by six live
training runs, and a crash-restart would pick up an edit mid-experiment.

    python scripts/sorted_exec.py --device cuda:0
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

from flexuf.backbone.decoder import patchify  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402


def med(fn, warmup=10, iters=40):
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
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/sorted_exec_BEST.json")
    a = ap.parse_args(argv)
    # torch.cuda.Event is created on the CURRENT device and a bare
    # torch.cuda.synchronize() syncs the current device. Timing work on another
    # card returns numbers rather than an error, and they are wrong.
    torch.cuda.set_device(a.device)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    dec, K, j, Fp = net.dec, cfg.num_exits, cfg.split_depth, cfg.feature_patch

    curve = json.loads((ROOT / "results/curve_BEST.json").read_text())
    W = (a.width + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    H = (a.height + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    n = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)

    rows = []
    print(f"  qp {a.qp}, {n} tiles, {H}x{W}\n")
    print(f"  {'budget':>8}{'current':>11}{'sorted':>11}{'saved':>10}{'match':>8}")
    with torch.no_grad():
        x = torch.zeros(1, 3, H, W, device=dev)
        qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
        y, q, _ = net._encode_to_latent(x, qp)
        feat = dec.upsample(y)
        for g in range(j):
            feat = dec.groups[g](feat)
        tiles, nh, nw = patchify(feat, Fp)

        for bud in a.budgets:
            h = min((r for r in curve["rows"]
                     if r["qp"] == a.qp and r.get("hist")),
                    key=lambda r: abs(r.get("db_vs_uf_per_frame", 9) - bud))["hist"]
            p = torch.tensor(h, dtype=torch.float) / sum(h)
            cnt = (p * n).round().long()
            cnt[-1] += n - cnt.sum()
            em = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                            for k, c in enumerate(cnt)]).to(dev).clamp(min=j)

            def current():
                """What the decoder does today: mask, gather, scatter per group."""
                active = torch.arange(n, device=dev)
                work = tiles
                canvas = torch.zeros_like(tiles)
                for g in range(j, K):
                    work = dec.groups[g](work)
                    keep = em[active] > g
                    done = active[~keep]
                    if done.numel():
                        canvas[done] = work[~keep]
                    active, work = active[keep], work[keep]
                    if active.numel() == 0:
                        break
                return canvas

            # Sort once. Descending, so deeper tiles come first and "still
            # active at group g" is a prefix. The boundaries come from one
            # cumulative count, not from a mask per group.
            order = torch.argsort(em, descending=True, stable=True)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(n, device=dev)
            # counts[g] = how many tiles still run at group g. One host read.
            bounds = [int((em > g).sum().item()) for g in range(j, K)]
            sorted_tiles = tiles[order]

            def sorted_exec():
                work = sorted_tiles
                canvas = torch.zeros_like(sorted_tiles)
                lo = n
                for idx, g in enumerate(range(j, K)):
                    work = dec.groups[g](work)
                    hi = bounds[idx]          # tiles continuing past this group
                    if hi < lo:
                        canvas[hi:lo] = work[hi:lo]
                        work = work[:hi]      # a VIEW, no gather
                        lo = hi
                    if hi == 0:
                        break
                return canvas[inv]

            t_cur = med(current)
            t_srt = med(sorted_exec)
            # The two must produce the same pixels, or the speedup is fictional.
            same = torch.allclose(current(), sorted_exec(), atol=0, rtol=0)
            rows.append({"budget_db": bud, "ms_current": t_cur,
                         "ms_sorted": t_srt, "identical": bool(same),
                         "hist": h})
            print(f"  {bud:>7.1f}{t_cur:>10.2f}ms{t_srt:>10.2f}ms"
                  f"{100*(1-t_srt/t_cur):>9.1f}%{'yes' if same else 'NO':>8}")

    print("\n  saved = of the per-tile group loop only; the stem, head and seam")
    print("  repair are untouched by this change.")
    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "qp": a.qp, "n_tiles": n,
         "device": torch.cuda.get_device_name(dev), "rows": rows}, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
