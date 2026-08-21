"""The seam penalty across the whole QP range, and what training does to it.

Three questions this answers, all of which were asked and none of which the
existing tables could settle:

1. The padding ablation was run at qp 0 / 32 / 63 only. Three points cannot
   show the shape of a curve, and the seam is the one quantity in this project
   that grows fastest with rate -- exactly where the method is weakest.

2. Is the seam INSIDE the 0.1 dB budget the routed results are quoted at?
   It is. Every routed number uses the RELEASE decoder's FULL-FRAME decode of
   the same latent as its reference, and our side is tiled, so the seam is
   charged. This script measures the same reference so the two are comparable.

3. Then why is the budget only 0.1 dB when replicate padding alone measured
   0.107 dB at qp 63? Because that table was measured on the WARM START --
   `runs/warmstart/ckpt_warmstart.pth.tar`, adapters zero-init, nothing
   trained. It is the raw geometric penalty before any weight has learned to
   live with it. The trained floor is the third curve here, on the same frames,
   with the same reference, at the same rates.

Metric is `psnr_611_420` throughout -- the 6:1:1 weighted YUV 4:2:0 PSNR that
DCVC-UF reports -- and it is a PER-FRAME average over the CTC set, not a single
image. The RGB-MSE-ratio convention that `paper_curve.py` uses for the routed
curve is computed alongside, so the two can be compared instead of assumed
equal.
"""
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
from ckpt import pinned as _pin  # noqa: E402
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=_pin("runs/BEST/ckpt_eval.pth.tar"),
                help="the TRAINED run whose floor is the third curve")
ap.add_argument("--qps", type=int, nargs="+",
                default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--max_seqs", type=int, default=0,
                help="cap the test set (0 = all present). Only for reproducing "
                     "an older table: the seam ablation in docs/07-seam.md was "
                     "measured when 10 sequences were on disk, and there are 40 "
                     "now. --max_seqs 10 --frames 2 --spread reproduces it.")
ap.add_argument("--seqs", nargs="*", default=None,
                help="select sequences BY NAME. --max_seqs 10 is not a control "
                     "for the old table: discovery order changed when MCL-JCV "
                     "landed, so the first ten today are not the first ten "
                     "then. Pass the names the old JSON recorded.")
ap.add_argument("--spread", action="store_true",
                help="sample frames across the sequence instead of taking the "
                     "first, as ctc_seam_ablation.py does")
ap.add_argument("--device", default=_gpu("cuda:0"))
ap.add_argument("--out", default="results/seam_vs_qp.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg_t = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
warm_path = reference_for(cfg_t, None)
warm = torch.load(warm_path, map_location="cpu", weights_only=False)

# Warm-start ladder, no repair module, one instance per padding rule. Same
# weights in both -- padding is a property of the config, not of the file.
def warm_model(pad):
    c = FlexUFConfig(**{**cfg_t.__dict__, "tile_pad_mode": pad,
                        "seam_repair": "none"})
    m = FlexUFIntra(c).to(dev).eval(); load_flexuf_state(m, warm)
    return m

rel = warm_model("replicate")          # its forward_full IS the release decode
m_zero, m_repl = warm_model("zeros"), rel
trained = FlexUFIntra(cfg_t).to(dev).eval(); load_flexuf_state(trained, ck)

# The comparison is only "the seam and nothing else" if the latent is shared,
# which requires the encoders to be bit-identical. Asserted, not assumed.
sa, sb = rel.enc.state_dict(), trained.enc.state_dict()
assert max((sa[k] - sb[k]).abs().max().item() for k in sa) == 0.0

seqs, missing = C.discover([])
if a.seqs:
    want = set(a.seqs)
    seqs = [s for s in seqs if s["name"] in want]
    got = {s["name"] for s in seqs}
    assert got == want, f"not on disk: {sorted(want - got)}"
elif a.max_seqs:
    seqs = seqs[:a.max_seqs]
frames = []
for s in seqs:
    stride = max(1, s["frames"] // max(a.frames, 1)) if a.spread else 1
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, stride)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append((x[i:i+1], pl[i]))
print(f"  {len(frames)} CTC karesi / {len(seqs)} sekans, {len(missing)} yok, "
      f"{cfg_t.rgb_patch}px tile, j={cfg_t.split_depth}, "
      f"repair={cfg_t.seam_repair}\n", flush=True)

P = cfg_t.rgb_patch
hdr = ("qp", "zeros", "replicate", "trained", "trained_rgb")
print("  " + "".join(f"{h:>13}" for h in hdr))
rows = []
with torch.no_grad():
    for qp_v in a.qps:
        acc = dict(zeros=0.0, replicate=0.0, trained=0.0, trained_rgb=0.0)
        for x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = rel._encode_to_latent(xp, qp)
            nt = ((H + ph) // P) * ((W + pw) // P)
            em = torch.full((nt,), cfg_t.num_exits - 1, device=dev)

            base = rel.dec.forward_full(y, q)          # released, full-frame
            d = C.psnr_611_420(base[:, :, :H, :W], pl)
            for name, m in (("zeros", m_zero), ("replicate", m_repl),
                            ("trained", trained)):
                o = m.dec(y, q, exit_map=em)
                acc[name] += d - C.psnr_611_420(o[:, :, :H, :W], pl)
            # Same trained decode, the other convention: 10log10 of the RGB MSE
            # ratio on the padded frame, which is what the routed curve reports.
            o = trained.dec(y, q, exit_map=em)
            mo = ((o - xp) ** 2).mean(); mb = ((base - xp) ** 2).mean()
            acc["trained_rgb"] += (10 * torch.log10(mo / mb)).item()
        n = len(frames)
        r = {"qp": qp_v, **{k: v / n for k, v in acc.items()}}
        rows.append(r)
        print("  " + f"{qp_v:>13}" +
              "".join(f"{r[k]:>13.4f}" for k in hdr[1:]), flush=True)

Path("results").mkdir(exist_ok=True)
json.dump({"ckpt": a.ckpt, "ref": str(warm_path), "n_frames": len(frames),
           "tile_px": P, "split_depth": cfg_t.split_depth,
           "metric": "psnr_611_420, per-frame mean, dB BELOW the release",
           "sequences": [s["name"] for s in seqs], "rows": rows},
          open(a.out, "w"), indent=2)
print(f"\n  -> {a.out}")
