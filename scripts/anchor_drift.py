"""How far has our deepest exit drifted from the RELEASED DCVC-UF decoder?

Every saving figure this project reports is "x% cheaper at y dB". The y is
measured against our own deepest exit. That is only a fair reference if the
deepest exit still IS DCVC-UF -- which it was, bit-exactly, at warm-start, but
training the whole decoder can move it. If it drifted down by 0.3 dB, then
"37% at 0.1 dB" is really 37% at 0.4 dB against the thing a reviewer would
compare us to, and the headline is wrong by a factor of four.

The comparison is exact by construction: --freeze_encoder means the encoder,
hyperprior and entropy model are untouched, so the warm-start checkpoint and the
trained one produce the IDENTICAL latent from the identical bitstream. Only the
synthesis differs, and the gap is the drift and nothing else.
"""
import argparse, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--device", default="cuda:4")
a = ap.parse_args()

dev = a.device
base = torch.load("runs/warmstart/ckpt_warmstart.pth.tar", map_location="cpu", weights_only=False)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()

stock = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(stock, base)
ours = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(ours, ck)

# The encoders must be identical, or the latents differ and the comparison is
# not the one claimed. Asserted, not assumed.
sd_s, sd_o = stock.enc.state_dict(), ours.enc.state_dict()
assert set(sd_s) == set(sd_o), "encoder anahtarlari farkli"
worst = max((sd_s[k] - sd_o[k]).abs().max().item() for k in sd_s)
print(f"  encoder ayni mi                      max|diff| = {worst}")
assert worst == 0.0, "encoder degismis -- latent ayni degil, kiyas gecersiz"

seqs, _ = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        frames.append((s["cls"], x[0:1], pl[0]))
print(f"\n  {len(frames)} CTC karesi\n")
print(f"  {'qp':>4}{'stok UF':>10}{'bizim en derin':>16}{'kayma':>9}")
for qp_v in a.qps:
    ds = do = 0.0
    with torch.no_grad():
        for _, x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % cfg.rgb_patch, (-W) % cfg.rgb_patch
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = stock._encode_to_latent(xp, qp)
            ds += C.psnr_611_420(stock.dec.forward_full(y, q)[:, :, :H, :W], pl)
            do += C.psnr_611_420(ours.dec.forward_full(y, q)[:, :, :H, :W], pl)
    n = len(frames)
    print(f"  {qp_v:>4}{ds/n:>10.3f}{do/n:>16.3f}{do/n - ds/n:>+9.3f}", flush=True)
