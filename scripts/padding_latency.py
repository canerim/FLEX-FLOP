"""Wall-clock cost of each tile-padding scheme.

The arls paper's own caveat was "a moderate increase in time cost", and this
project's entire claim is about compute. Adopting arls as the default while
measuring only its dB benefit would be claiming a free win without checking the
bill -- exactly the accounting error that made the early saving figures too
good. So the padding is timed on the path it actually runs on: a per-tile decode
of a 1080p frame.

MAC count would not answer this. The AR fit is two reductions per tile per
block, negligible in MACs, but it adds kernel launches and a synchronisation-
free but memory-bound pass, and on a GPU that is what costs time.
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
from flexuf.config import FlexUFConfig, LATENT_CH, TRUNK_CH
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.backbone.padding import wrap_tile_padding

dev = "cuda:6"
ck = torch.load("runs/warmstart/ckpt_warmstart.pth.tar", map_location="cpu", weights_only=False)
print(f"  {'mod':<11}{'j2/128px':>12}{'j2/256px':>12}   (ms, 1080p tam decode, deepest exit)")
res = {}
for mode in ("zeros", "replicate", "linear", "arls"):
    row = []
    for lp in (8, 16):
        cfg = FlexUFConfig(split_depth=2, latent_patch=lp, tile_pad_mode="zeros",
                           seam_repair="none")
        net = FlexUFIntra(cfg).to(dev).eval()
        load_flexuf_state(net, ck)
        undo = wrap_tile_padding(net.dec.groups, cfg.split_depth, mode)
        # 1080p padded up to a whole number of tiles, which differs per tile
        # size: 1920x1152 for 128px tiles, 2048x1280 for 256px. Padding to 1152
        # with 256px tiles leaves 4.5 tile rows -- the assert that caught this.
        H = -(-1080 // cfg.rgb_patch) * cfg.rgb_patch
        W = -(-1920 // cfg.rgb_patch) * cfg.rgb_patch
        y = torch.randn(1, LATENT_CH, H // 16, W // 16, device=dev)
        q = torch.rand(1, TRUNK_CH, 1, 1, device=dev) + 0.5
        nt = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)
        em = torch.full((nt,), cfg.num_exits - 1, device=dev)
        with torch.no_grad():
            for _ in range(3):
                net.dec(y, q, exit_map=em)
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            for _ in range(10):
                net.dec(y, q, exit_map=em)
            torch.cuda.synchronize(dev)
            row.append((time.perf_counter() - t) / 10 * 1000)
        undo(); del net; torch.cuda.empty_cache()
    res[mode] = row
    extra = "" if mode == "zeros" else "   " + "  ".join(
        f"{100*(row[i]/res['zeros'][i]-1):+5.1f}%" for i in range(2))
    print(f"  {mode:<11}" + "".join(f"{v:>12.1f}" for v in row) + extra, flush=True)
