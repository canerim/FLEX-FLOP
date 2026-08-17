"""The reference every saving is quoted against IS Microsoft's released decoder.

Every number in this project is "x% saved at y dB below DCVC-UF", and all of
them compare against `runs/warmstart/ckpt_warmstart.pth.tar` -- the released
weights re-expressed in ladder form. That file is one remap away from the
release, and if the remap were wrong every result would be wrong in a way no
downstream check could catch: the frontier, the BD numbers and the target
verdicts would all be internally consistent and measured against the wrong
decoder.

The warm-start builder asserts bit-exactness on the tensors. This asserts it on
the OUTPUT, through Microsoft's own class and their own forward function, so
the claim does not rest on the remap being audited correctly -- it rests on the
two models producing the same pixels.

    python tests/test_reference_is_the_release.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

RELEASE = Path.home() / "DCVC" / "checkpoints" / "cvpr2026_image.pth.tar"
WARMSTART = ROOT / "runs" / "warmstart" / "ckpt_warmstart.pth.tar"


def test_deepest_exit_is_the_released_decoder(qp):
    from src.models.image_model import DMCI
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    rel = DMCI()
    sd = torch.load(RELEASE, map_location="cpu", weights_only=False)
    rel.load_state_dict(sd.get("state_dict", sd))
    rel = rel.to(dev).eval()

    ck = torch.load(WARMSTART, map_location="cpu", weights_only=False)
    ours = FlexUFIntra(FlexUFConfig(**ck["config"])).to(dev).eval()
    load_flexuf_state(ours, ck)

    # A fixed random frame rather than a photograph: the claim is about the
    # weights, and content that happens to be easy could hide a small mismatch.
    torch.manual_seed(0)
    x = (torch.rand(1, 3, 512, 512, device=dev) - 0.5)

    with torch.no_grad():
        q = torch.full((1,), qp, dtype=torch.int32, device=dev)
        theirs = rel.forward_one_frame(x, q)["x_hat"]
        y, qd, _ = ours._encode_to_latent(x, q)
        mine = ours.dec.forward_full(y, qd)

    assert (theirs - mine).abs().max().item() == 0.0, (
        f"the reference is NOT the released decoder at qp{qp}; every saving "
        f"quoted against it is measured against something else")


if __name__ == "__main__":
    if not RELEASE.exists() or not WARMSTART.exists():
        raise SystemExit(f"needs {RELEASE} and {WARMSTART}")
    print("the reference every saving is quoted against, vs Microsoft's release:")
    for _qp in (0, 32, 63):
        test_deepest_exit_is_the_released_decoder(_qp)
        print(f"  qp{_qp:<3} max|x_hat diff| = 0.0")
    print("\nthe reference IS the released decoder, through their own class and "
          "forward.\n")
