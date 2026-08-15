"""Build a multi-exit model on top of Microsoft's released DCVC-UF-Intra weights.

Why this is the important path, not a convenience
-------------------------------------------------
Training from scratch gave a ladder that is not monotonic through the deployed
decode: measured on e1 at epoch 2, exit 3 lost 0.2137 dB while exit 2 lost only
0.1787 dB — the deeper exit costing 15 points more compute and returning *worse*
quality. Routing means spending more where it buys more, so that trade does not
exist and a uniform depth wins. It is not a router failure; the ladder is not
ready.

FLEX-FLOP hit exactly this and the released weights are what got it past. With
them the deepest exit *is* the reference codec from step zero — bit-exact, not
approximately — so ordering is guaranteed by construction and training only has
to learn the shallow exits' corrections.

What this script does
---------------------
1. Verifies the release loads cleanly into stock DMCI (nothing missing, nothing
   unexpected) — a partial or mismatched checkpoint must fail loudly here rather
   than silently produce a half-random codec.
2. Builds FlexUFIntra and transfers: encoder / hyperprior / entropy model /
   q_scale tables verbatim, decoder through the ladder remap.
3. Asserts the deepest exit reproduces stock UF bit-exactly (max|diff| == 0).
4. Asserts every untrained exit equals stock UF truncated at that depth, which
   is what the zero-init adapters guarantee.
5. Saves runs/warmstart/ckpt_warmstart.pth.tar, ready for adapter training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.image_model import DMCI, IntraDecoder  # noqa: E402

from flexuf.backbone.warmstart import remap_decoder_state  # noqa: E402
from flexuf.config import LATENT_CH, TRUNK_CH, FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra  # noqa: E402

RELEASE = DCVC_ROOT / "checkpoints" / "cvpr2026_image.pth.tar"
OUT = Path.home() / "FLEX-UF" / "runs" / "warmstart"


def main() -> int:
    print(f"release: {RELEASE}")
    ck = torch.load(RELEASE, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    print(f"  {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")

    # ---- 1. it must be a complete DMCI ----------------------------------
    stock = DMCI()
    missing, unexpected = stock.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  FATAL: not a clean DMCI checkpoint — "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        print(f"    missing e.g. {list(missing)[:4]}")
        print(f"    unexpected e.g. {list(unexpected)[:4]}")
        return 1
    print("  loads cleanly into stock DMCI")

    # ---- 2. transfer into the ladder ------------------------------------
    cfg = FlexUFConfig(split_depth=2, latent_patch=8, latent_halo=2,
                       adapter_kind="conv1x1")
    net = FlexUFIntra(cfg)

    non_dec = {k: v for k, v in sd.items() if not k.startswith("dec.")}
    dec_src = {k[len("dec."):]: v for k, v in sd.items() if k.startswith("dec.")}
    remapped, unknown = remap_decoder_state(dec_src, cfg.blocks_per_exit)
    if unknown:
        print(f"  FATAL: decoder remap did not understand {unknown[:4]}")
        return 1

    full = dict(non_dec)
    full.update({f"dec.{k}": v for k, v in remapped.items()})
    miss, unexp = net.load_state_dict(full, strict=False)
    bad = [k for k in miss if "adapters" not in k]
    if bad or unexp:
        print(f"  FATAL: transfer incomplete — missing {bad[:4]} unexpected {list(unexp)[:4]}")
        return 1
    print(f"  transferred {len(full)} tensors; "
          f"{len(miss)} adapter tensors left at zero-init (by design)")

    # ---- 3+4. the controls ----------------------------------------------
    net.eval()
    ref = IntraDecoder().eval()
    ref.load_state_dict(dec_src)
    y = torch.randn(1, LATENT_CH, 16, 16)
    q = torch.rand(1, TRUNK_CH, 1, 1) + 0.5

    with torch.no_grad():
        deepest = (ref(y, q) - net.dec.forward_full(y, q)).abs().max().item()
        worst = 0.0
        b = cfg.blocks_per_exit
        for k in range(cfg.num_exits):
            f = ref.dec_1[0](y)
            for n in range(1, (k + 1) * b + 1):
                f = ref.dec_1[n](f)
            r = torch.nn.functional.pixel_shuffle(ref.dec_2(f * q), 8)
            worst = max(worst, (r - net.dec.forward_full(y, q, exit_idx=k)).abs().max().item())

    print(f"  CONTROL deepest exit vs released UF : max|diff| = {deepest}")
    print(f"  CONTROL every exit vs truncated UF  : max|diff| = {worst}")
    if deepest != 0.0 or worst != 0.0:
        print("  FATAL: warm-start is not bit-exact; refusing to save")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ckpt_warmstart.pth.tar"
    torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__,
                "provenance": "microsoft cvpr2026_image.pth.tar, ladder remap"}, path)
    (OUT / "warmstart_report.json").write_text(json.dumps({
        "release": str(RELEASE),
        "tensors_transferred": len(full),
        "adapters_at_init": len(miss),
        "control_deepest_max_diff": deepest,
        "control_all_exits_max_diff": worst,
        "config": cfg.__dict__,
    }, indent=2))
    print(f"\n  saved {path}")
    print("  the deepest exit is now bit-exact released DCVC-UF, so the ladder is")
    print("  ordered by construction — the failure mode that sank the from-scratch")
    print("  run cannot occur here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
