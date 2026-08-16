"""Does the MAC-based saving we report actually show up on the clock?

The project's headline is a percentage. That percentage comes from `flexuf/cost.py`,
which counts MACs. MACs are the right unit for a paper and the wrong unit for a
promise: a decoder that skips 40% of its multiply-accumulates but only runs 15%
faster has not saved 40% of anything a user can feel.

This is the same question that caught arls -- benefit measured in one unit, bill
paid in another -- asked about the headline itself rather than about a component.

Method: decode one 1080p frame with every tile at exit k, for each k, and compare
the measured time ratio against exit_costs(). No routing, no data: the point is
the mapping from depth to time, and a router only picks points on this curve.
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
from flexuf.config import FlexUFConfig, LATENT_CH, TRUNK_CH
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.cost import exit_costs

dev = "cuda:6"
ck = torch.load("runs/warmstart/ckpt_warmstart.pth.tar", map_location="cpu", weights_only=False)
for lp, name in ((8, "j2 / 128px"), (16, "j2 / 256px")):
    cfg = FlexUFConfig(split_depth=2, latent_patch=lp, tile_pad_mode="replicate",
                       seam_repair="grid")
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    H = -(-1080 // cfg.rgb_patch) * cfg.rgb_patch
    W = -(-1920 // cfg.rgb_patch) * cfg.rgb_patch
    y = torch.randn(1, LATENT_CH, H // 16, W // 16, device=dev)
    q = torch.rand(1, TRUNK_CH, 1, 1, device=dev) + 0.5
    nt = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)
    C = exit_costs(cfg, "head").tolist()

    def t(em):
        with torch.no_grad():
            for _ in range(3):
                net.dec(y, q, exit_map=em)
            torch.cuda.synchronize(dev); s = time.perf_counter()
            for _ in range(10):
                net.dec(y, q, exit_map=em)
            torch.cuda.synchronize(dev)
        return (time.perf_counter() - s) / 10 * 1000

    full = t(torch.full((nt,), cfg.num_exits - 1, device=dev))
    print(f"\n  {name}  ({nt} tile, en derin cikis {full:.1f} ms)")
    print(f"    {'cikis':>6}{'MAC tasarrufu':>15}{'saat tasarrufu':>16}{'fark':>8}")
    for k in range(cfg.split_depth, cfg.num_exits):
        ms = t(torch.full((nt,), k, device=dev))
        mac = 100 * (1 - C[k] / C[-1])
        wall = 100 * (1 - ms / full)
        print(f"    {k:>6}{mac:>14.1f}%{wall:>15.1f}%{wall-mac:>+8.1f}")
    del net; torch.cuda.empty_cache()
