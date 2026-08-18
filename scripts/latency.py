"""Does the MAC saving become a wall-clock saving? Measured, not assumed.

This is the highest-severity open risk in docs/06-cvpr-plan.md. Every saving in
this project is a MAC count, and the field this work would be published in
measures frames per second -- DCVC-UF's own paper spends half of Table 3 on it.
A MAC count cannot see kernel-launch overhead, gather/scatter of tiles,
patchify/unpatchify, the seam-repair pass, or the drop in arithmetic intensity
when a group runs on 300 tiles instead of 1500.

The decoder already executes the efficient way -- a shrinking active set, where
each group runs only on the tiles still climbing the ladder -- so this measures
the implementation as it stands rather than a strawman.

Three timings, all on the same latent so nothing but the synthesis differs:

  stock          the released decoder's forward_full: the number to beat
  ladder, deep   our tiled forward with every tile at the deepest exit. This is
                 the OVERHEAD FLOOR: same arithmetic as stock plus tiling, so
                 whatever it costs is what the machinery costs before routing
                 has saved anything.
  ladder, routed our tiled forward with the exit map the system actually picks
                 at the 0.1 dB budget, read from results/curve_BEST.json.

Reported against the MAC model's prediction, so the gap between predicted and
realised is explicit rather than buried.

    python scripts/latency.py --ckpt runs/BEST/ckpt_eval.pth.tar
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

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs, frame_relative_cost  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402


def timeit_interleaved(fns: dict, warmup: int, iters: int) -> dict:
    """Median ms for each variant, measured INTERLEAVED.

    Timing 30 of A, then 30 of B, then 30 of C was the first version, and it is
    wrong on a shared card: this GPU sits at 100% with a training job on it, and
    any drift in that job's load between the three blocks lands entirely in the
    ratio -- which is the only number this script exists to produce. Running one
    iteration of each per round spreads the drift across all variants instead.

    Median rather than mean, for the same reason: one stall should not move the
    result.
    """
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    torch.cuda.synchronize()
    ts = {k: [] for k in fns}
    for _ in range(iters):
        for k, fn in fns.items():
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts[k].append(a.elapsed_time(b))
    return {k: statistics.median(v) for k, v in ts.items()}


def _content(H, W, dev):
    """A deterministic non-uniform image.

    `torch.zeros` was used here and in the sorted-execution prototype, which made
    every tile identical -- so a check that reordering tiles changes nothing
    could not fail, and every tile presented the encoder with the same trivial
    content. Smooth gradients plus a coarse texture give the entropy model
    something to do and the tiles something to differ about, without depending on
    a file being present.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    yy = torch.linspace(0, 1, H).view(1, 1, H, 1)
    xx = torch.linspace(0, 1, W).view(1, 1, 1, W)
    base = (yy * xx).expand(1, 3, H, W).clone()
    tex = torch.rand(1, 3, H // 16, W // 16, generator=g)
    tex = torch.nn.functional.interpolate(tex, size=(H, W), mode="bilinear",
                                          align_corners=False)
    return (0.6 * base + 0.4 * tex - 0.5).to(dev)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/BEST/ckpt_eval.pth.tar")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget_db", type=float, default=0.1,
                    help="which operating point's exit map to time. The fixed "
                         "tiling overhead is the same at every budget, so a "
                         "looser budget amortises it over a larger saving -- "
                         "which is a claim worth measuring, not assuming.")
    ap.add_argument("--sorted_tiles", action="store_true",
                   help="time the sorted per-tile loop as a fourth variant. The "
                        "arithmetic is identical (tests/test_sorted_tiles.py); "
                        "what changes is one device-to-host synchronisation per "
                        "group boundary becoming one for the whole loop.")
    ap.add_argument("--out", default="results/latency_BEST.json")
    ap.add_argument("--note", default="",
                    help="recorded with the result; use it to say what else was "
                         "on the card, because that decides how much the "
                         "numbers mean")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    C = exit_costs(cfg, "head")

    # The exit maps the system actually chooses, from the measured frontier.
    curve = json.loads((ROOT / "results/curve_BEST.json").read_text())

    def hist_at(qp, db=0.1):
        rows = [r for r in curve["rows"] if r["qp"] == qp and r.get("hist")]
        return min(rows, key=lambda r: abs(
            r.get("db_vs_uf_per_frame", 9) - db))["hist"]

    W = (a.width + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    H = (a.height + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    print(f"  {a.width}x{a.height} padded to {W}x{H}, {cfg.rgb_patch}px tiles, "
          f"exit map at the {a.budget_db:g} dB point, "
          f"{a.iters} iters after {a.warmup} warmup, median\n")

    rows = []
    print(f"  {'qp':>4}{'stock':>10}{'deep':>10}{'routed':>10}"
          f"{'overhead':>11}{'realised':>11}{'predicted':>11}"
          + (f"{'sorted':>11}" if a.sorted_tiles else ""))
    with torch.no_grad():
        for qp_v in a.qps:
            # Real content, not zeros. With a constant image every tile carries
            # the same latent, so anything that reorders tiles is trivially
            # exact and any data-dependent kernel choice is unrepresentative.
            x = _content(H, W, dev)
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(x, qp)

            n_tiles = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)
            h = hist_at(qp_v, a.budget_db)
            # Rebuild a per-tile exit map with the measured proportions.
            props = torch.tensor(h, dtype=torch.float) / sum(h)
            counts = (props * n_tiles).round().long()
            counts[-1] += n_tiles - counts.sum()
            em = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                            for k, c in enumerate(counts)]).to(dev)
            deep = torch.full((n_tiles,), cfg.num_exits - 1,
                              dtype=torch.long, device=dev)

            variants = {
                "stock": lambda: ref.dec.forward_full(y, q),
                "deep": lambda: net.dec(y, q, exit_map=deep),
                "routed": lambda: net.dec(y, q, exit_map=em)}
            if a.sorted_tiles:
                cfg_s = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": True})
                cfg_m = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": False})

                def _routed_sorted():
                    net.dec.cfg = cfg_s
                    try:
                        return net.dec(y, q, exit_map=em)
                    finally:
                        net.dec.cfg = cfg_m
                variants["routed_sorted"] = _routed_sorted
            t = timeit_interleaved(variants, a.warmup, a.iters)
            t_stock, t_deep, t_route = t["stock"], t["deep"], t["routed"]
            t_sort = t.get("routed_sorted")

            predicted = 100 * (1 - frame_relative_cost(em.cpu(), cfg, "head"))
            realised = 100 * (1 - t_route / t_stock)
            overhead = 100 * (t_deep / t_stock - 1)
            row = {"qp": qp_v, "ms_stock": t_stock, "ms_deep": t_deep,
                   "ms_routed": t_route, "overhead_pct": overhead,
                   "realised_saving_pct": realised,
                   "predicted_saving_pct": predicted, "hist": h}
            line = (f"  {qp_v:>4}{t_stock:>9.2f}ms{t_deep:>9.2f}ms"
                    f"{t_route:>9.2f}ms{overhead:>10.1f}%{realised:>10.1f}%"
                    f"{predicted:>10.1f}%")
            if t_sort is not None:
                row["ms_routed_sorted"] = t_sort
                row["realised_saving_sorted_pct"] = 100 * (1 - t_sort / t_stock)
                line += f"{100*(1-t_sort/t_stock):>10.1f}%"
            rows.append(row)
            print(line)

    print("\n  overhead = what the tiling machinery costs before routing saves")
    print("  anything (every tile at the deepest exit, same arithmetic as stock)")
    print("  realised = wall-clock; predicted = the MAC model. The gap between")
    print("  them is the number this project has never measured.")
    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "resolution": [a.width, a.height],
         "padded": [W, H], "tile_px": cfg.rgb_patch,
         "device": torch.cuda.get_device_name(dev),
         "budget_db": a.budget_db,
         "warmup": a.warmup, "iters": a.iters, "interleaved": True,
         "note": a.note, "rows": rows}, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
