"""Per-exit quality on HELD-OUT OpenImages, to separate two explanations.

CONTROL's exits degraded over its second epoch when measured on CTC video, while
its training-batch spread did not move. Two readings fit that:

  (a) the shallow exits are losing generality -- they fit the training
      distribution better and everything else worse;
  (b) they are fine on the training distribution including unseen samples from
      it, and what they lost is specifically the transfer from 512px OpenImages
      photographs to 1080p video frames.

The dataset ships description_val.json: 512 images, zero overlap with the
379,614 in description.json. Measuring there is the same distribution the model
trains on, without the images it saw. Comparing that against the CTC numbers
separates (a) from (b), which the training log cannot because it only ever sees
images the model is currently learning from.

Same protocol as why_qp.py otherwise: identical latents from the shared encoder,
distortion against the released decoder, so the only difference is synthesis.

    python scripts/why_qp_val.py --ckpt runs/CONTROL/ckpt_epo0.pth.tar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402

VAL = Path("/data10/shareddata/openimages/dcvc_train")


def load_val(n: int, crop: int, device):
    """`n` held-out images, centre-cropped to a multiple of the tile size."""
    from PIL import Image
    import numpy as np
    # The training pipeline converts RGB to YCbCr before subtracting 0.5
    # (~/DCVC/src/datasets/image_dataset.py). Feeding raw RGB gives the model a
    # different colour space than it was trained in, and the numbers come out
    # meaningless -- the warm start's exit 2 read -0.24 dB where the validated
    # CTC path says +2.80 for the same checkpoint. Same class of mistake as
    # plotting YCbCr as RGB in the example-frame figure.
    from src.utils.transforms import rgb2ycbcr_np
    names = list(json.loads((VAL / "description_val.json").read_text()))[:n]
    out = []
    for nm in names:
        p = VAL / nm
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w < crop or h < crop:
            continue
        left, top = (w - crop) // 2, (h - crop) // 2
        a = np.asarray(im.crop((left, top, left + crop, top + crop)),
                       dtype=np.float32) / 255.0
        a = rgb2ycbcr_np(a) - 0.5
        out.append(torch.from_numpy(a).permute(2, 0, 1)[None].to(device))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    assert max((sa[k] - sb[k]).abs().max().item() for k in sa) == 0.0

    imgs = load_val(a.images, a.crop, dev)
    if not imgs:
        raise SystemExit("no usable validation images")
    print(f"  {len(imgs)} held-out OpenImages at {a.crop}px, "
          f"{cfg.rgb_patch}px tile\n")
    print(f"  {'qp':>4}" + "".join(f"{'exit ' + str(k):>10}"
                                   for k in range(cfg.num_exits)))

    rows = []
    K = cfg.num_exits
    with torch.no_grad():
        for qp_v in a.qps:
            se = torch.zeros(K, device=dev)
            sr = torch.zeros((), device=dev)
            for x in imgs:
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(x, qp)
                for i, o in enumerate(net.dec.forward_all_exits(y, q)):
                    se[i] += ((o - x) ** 2).mean()
                sr += ((ref.dec.forward_full(y, q) - x) ** 2).mean()
            db = (10 * torch.log10(se / sr)).tolist()
            rows.append({"qp": qp_v, "db_per_exit": db})
            print(f"  {qp_v:>4}" + "".join(f"{v:>10.4f}" for v in db))

    if a.out:
        (ROOT / a.out).write_text(json.dumps(
            {"ckpt": a.ckpt, "n_images": len(imgs), "crop": a.crop,
             "split": "description_val.json", "rows": rows}, indent=2))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
