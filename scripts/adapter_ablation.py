"""What did the exit adapters buy?

Each shallow exit hands the shared head a feature the head was not fitted to, and
the adapter is the correction. The paper describes them and never measures them.
Zeroing an adapter restores the identity it was initialised to, so the trained
decoder can be evaluated with and without each one -- no retraining, and the
difference is what that adapter learned.

Two ablations:

  all off   every adapter set to the identity, all exits
  per exit  one adapter at a time, so the contribution is attributed

The deepest exit has no adapter by construction, and is the control: it must not
move.
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
from ckpt import pinned as _pin  # noqa: E402
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.eval import reference_frame_mse, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=_pin("runs/RECIPE512/ckpt_eval.pth.tar"))
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--max_seqs", type=int, default=16)
ap.add_argument("--device", default=_gpu("cuda:7"))
ap.add_argument("--out", default="results/adapter_ablation.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                  weights_only=False))
K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch
saved = copy.deepcopy(net.dec.adapters.state_dict())


def zero_adapters(which):
    """Set the LAST layer of each named adapter to zero -> exact identity."""
    net.dec.adapters.load_state_dict(saved)
    for i in which:
        ad = net.dec.adapters[i]
        last = getattr(ad, "conv", None) or getattr(ad, "pw_out", None)
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)


seqs, _ = C.discover([])
seqs = seqs[:a.max_seqs]
frames = []
for s in seqs:
    x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
    if x is not None:
        frames.append(x[0:1])
print(f"  {len(frames)} frames, {K} exits, adapters on 0..{K-2}\n", flush=True)

rows = []
with torch.no_grad():
    for qp_v in a.qps:
        cells = []
        for x in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            nt = ((H + ph) // P) * ((W + pw) // P)
            cells.append((y, q, xp, reference_frame_mse(ref.dec, y, q, xp), nt))

        def at_exit(e):
            t = 0.0
            for y, q, xp, R, nt in cells:
                em = torch.full((nt,), e, dtype=torch.long, device=dev)
                t += (10 * torch.log10(
                    true_frame_mse(net.dec, y, q, xp, em) / R)).item()
            return t / len(cells)

        zero_adapters([])
        base = {e: at_exit(e) for e in range(j, K)}
        zero_adapters(range(K - 1))
        alloff = {e: at_exit(e) for e in range(j, K)}
        per = {}
        for i in range(j, K - 1):
            zero_adapters([i])
            per[i] = at_exit(i)
        zero_adapters([])

        rows.append({"qp": qp_v, "trained": base, "all_off": alloff,
                     "one_off": per})
        print(f"  qp {qp_v}")
        print(f"    {'exit':>5}{'trained':>10}{'no adapter':>13}{'gain':>9}")
        for e in range(j, K):
            g = alloff[e] - base[e]
            tag = "  (control: no adapter exists)" if e == K - 1 else ""
            print(f"    {e:>5}{base[e]:>10.4f}{alloff[e]:>13.4f}{g:>+9.4f}{tag}")
        print(flush=True)

json.dump({"ckpt": a.ckpt, "n_frames": len(frames), "rows": rows},
          open(a.out, "w"), indent=2)
print(f"  -> {a.out}")
