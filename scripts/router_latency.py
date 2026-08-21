"""Does the router's 0.163% of MACs cost 0.163% of the time?

Section 5.8 of the paper is an argument that operations and wall-clock are
different currencies: a 35.3% cut in operations buys 29.1% of the time, and the
shortfall grows with the cut. Every configuration-B number in this work charges
the router at its MAC share. If that argument is right, the MAC share is the
wrong price.

Three timings on the same latent, interleaved so a co-tenant's load drift lands
on all of them equally:

  decode        the tiled decoder at the deepest exit -- what B is a fraction of
  stem          the first j groups, which B does NOT pay for: the decoder
                computes them anyway before the split
  router        the head alone, on the stem it was handed

The number that matters is `router / decode`, against the 0.163% the MAC model
predicts. A 144 K head is small enough that its cost is launch overhead rather
than arithmetic, and launch overhead is not in a MAC count.

    python scripts/router_latency.py --ckpt runs/RECIPE512/ckpt_eval.pth.tar \
        --router2 runs/RECIPE512/routers2/v2_lam1.3e-5.pth --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

from flexuf.config import FlexUFConfig, LATENT_CH, TRUNK_CH  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from latency import _content, timeit_interleaved  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--router2", required=True)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    # The deployed path pads 1920x1080 up to a multiple of the tile size, which
    # for 256 px tiles is 2048x1280 -- 8x5 = 40 tiles. Timing the unpadded frame
    # would time a shape the decoder never sees.
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/router_latency.json")
    a = ap.parse_args(argv)

    dev = a.device
    # torch.cuda.Event is created on the CURRENT device, not on the tensors'
    # device. Timing work on cuda:7 with events on cuda:0 returns ~0 ms, which
    # is how the first run of this script reported a router costing 0.001% of
    # the decode.
    torch.cuda.set_device(dev)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)

    from flexuf.router.head2 import StemRouterHeadV2
    h = torch.load(ROOT / a.router2, map_location="cpu", weights_only=False)
    head = StemRouterHeadV2(TRUNK_CH, LATENT_CH, cfg.num_exits,
                            min_exit=cfg.split_depth).to(dev).eval()
    head.load_state_dict(h.get("router_head_v2", h.get("state_dict", h)))
    params = sum(p.numel() for p in head.parameters())

    x = _content(a.height, a.width, dev)
    rows = []
    with torch.no_grad():
        for qp_v in a.qps:
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, aux = net._encode_to_latent(x, qp)
            sc = aux["scales_hat"]
            stem = net.dec.upsample(y)
            for g in range(cfg.split_depth):
                stem = net.dec.groups[g](stem)
            n_tiles = ((a.height // cfg.rgb_patch) * (a.width // cfg.rgb_patch))
            deep = torch.full((n_tiles,), cfg.num_exits - 1,
                              dtype=torch.long, device=dev)

            def decode():
                net.dec(y, q, exit_map=deep)

            def stem_only():
                s = net.dec.upsample(y)
                for g in range(cfg.split_depth):
                    s = net.dec.groups[g](s)

            def router():
                head(stem, y, sc, qp, cfg.feature_patch, cfg.latent_patch)

            t = timeit_interleaved({"decode": decode, "stem": stem_only,
                                    "router": router}, a.warmup, a.iters)
            share = 100 * t["router"] / t["decode"]
            rows.append({"qp": qp_v, "decode_ms": t["decode"],
                         "stem_ms": t["stem"], "router_ms": t["router"],
                         "router_share_pct_time": share})
            print(f"  qp {qp_v:>3}   decode {t['decode']:7.2f} ms   "
                  f"stem {t['stem']:6.2f} ms   router {t['router']:6.3f} ms   "
                  f"= {share:.3f}% of decode")

    mac_share = 0.16288874189707633
    med = sorted(r["router_share_pct_time"] for r in rows)[len(rows) // 2]
    print(f"\n  MAC model says {mac_share:.3f}% of the decode; "
          f"wall-clock says {med:.3f}%, a factor of {med/mac_share:.1f}")
    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "router2": a.router2, "params": params,
         "resolution": [a.height, a.width], "iters": a.iters,
         "device": torch.cuda.get_device_name(dev),
         "router_share_pct_macs": mac_share,
         "router_share_pct_time_median": med,
         "time_over_mac_factor": med / mac_share, "rows": rows}, indent=2))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
