"""Where every module sits, what shape it is, and what it costs.

The supplement's implementation section needs one grid that a reader can
re-implement the decoder from: module, the grid it runs on, its tensors, its
parameter count and its arithmetic per output pixel. Nothing in results/ carried
that, so the shapes were being quoted from the source files instead, which is
exactly the provenance gap section A closes everywhere else.

This reads the pinned checkpoint's state dict on the CPU and derives everything
from tensor shapes alone. No model is instantiated, no GPU is touched, no file
under runs/ is written.

    MAC per output pixel of a Conv2d with weight [O, I/g, kh, kw] is
    O * (I/g) * kh * kw, which is a property of the weight and of nothing else.

Run:  ./.venv/bin/python scripts/module_shapes.py
Writes: results/supp_module_shapes.json
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
CKPT = R / "runs/RECIPE512/ckpt_PAPER.pth.tar"
ROUTER = R / "runs/RECIPE512/routers2/v2_lam1.3e-5.pth"
OUT = R / "results/supp_module_shapes.json"


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def mac_per_px(name, shape):
    """MAC per OUTPUT pixel for a conv weight; per call for a linear weight."""
    if not name.endswith("weight"):
        return 0
    if len(shape) == 4:
        o, i, kh, kw = shape
        return int(o * i * kh * kw)
    if len(shape) == 2:
        return int(shape[0] * shape[1])
    return 0



# ---------------------------------------------------------------------------
# One grid for the whole decoder.
#
# Every operator except one runs on the FEATURE grid, which is 1/8 of the frame
# per side and so 1/64 of its pixels. The exception is the subpel convolution
# that opens the decoder: it runs on the LATENT grid, one quarter as many
# positions, and pixel-shuffles by 2 into the feature grid. Charging it per
# feature pixel therefore means dividing its weight cost by 4. Doing that makes
# every module comparable in one column, and the shares that fall out reproduce
# the hook audit in scripts/mac_audit.py to three decimals without importing it.
# ---------------------------------------------------------------------------
LATENT_GRID = {"dec.upsample.up.conv.0.weight": 4}


def per_feature_px(role):
    tot = 0.0
    for t in role["tensors"]:
        d = LATENT_GRID.get(t["name"], 1)
        tot += t["mac_per_out_px"] / d
    return tot


def collect(sd, prefix):
    tensors, params, mac = [], 0, 0
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        s = tuple(v.shape)
        params += int(v.numel())
        mac += mac_per_px(k, s)
        tensors.append({"name": k, "shape": list(s), "params": int(v.numel()),
                        "mac_per_out_px": mac_per_px(k, s)})
    role = {"prefix": prefix, "params": params, "mac_per_out_px": mac,
            "tensors": tensors}
    role["mac_per_feature_px"] = per_feature_px(role)
    return role


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    enc = sum(int(v.numel()) for k, v in sd.items() if not k.startswith("dec."))
    dec = sum(int(v.numel()) for k, v in sd.items() if k.startswith("dec."))

    roles = OrderedDict()
    roles["upsample"] = collect(sd, "dec.upsample.")
    for g in range(6):
        roles[f"groups.{g}"] = collect(sd, f"dec.groups.{g}.")
    roles["one_block"] = collect(sd, "dec.groups.0.0.")
    for a in range(5):
        roles[f"adapters.{a}"] = collect(sd, f"dec.adapters.{a}.")
    roles["seam_repair"] = collect(sd, "dec.seam_repair.")
    roles["head"] = collect(sd, "dec.head.")
    roles["router_head_v1_in_ckpt"] = collect(sd, "router_head.")

    # The decode's own total, on the feature grid, and every module's share of
    # it. Nothing here is calibrated against anything: it is the sum of the
    # weight shapes the checkpoint carries.
    total = (roles["upsample"]["mac_per_feature_px"]
             + 12 * roles["one_block"]["mac_per_feature_px"]
             + roles["head"]["mac_per_feature_px"])
    shares = {name: r["mac_per_feature_px"] / total for name, r in roles.items()}
    shares["trunk_12_blocks"] = 12 * roles["one_block"]["mac_per_feature_px"] / total

    rk = torch.load(ROUTER, map_location="cpu", weights_only=False)
    rsd = rk["router_head_v2"]
    router = {"file": str(ROUTER.relative_to(R)),
              "lam": rk.get("lam"), "trained_on": rk.get("ckpt"),
              "params": sum(int(v.numel()) for v in rsd.values()),
              "tensors": [{"name": k, "shape": list(v.shape),
                           "params": int(v.numel())} for k, v in rsd.items()]}

    out = {
        "what": "module by module: where it sits, its tensors, its parameters, "
                "and its multiply-accumulates per output pixel",
        "ckpt": "runs/RECIPE512/ckpt_PAPER.pth.tar",
        "ckpt_md5": md5(CKPT),
        "ckpt_epoch": ck.get("epoch"),
        "config": ck.get("config"),
        "derivation": "shapes and parameter counts read from the state dict; "
                      "MAC per output pixel is O*(I/g)*kh*kw from the weight "
                      "shape alone. No model instantiated, no GPU used.",
        "params_total": enc + dec,
        "params_encoder_side": enc,
        "params_decoder_side": dec,
        "grids": {
            "latent": "1/16 of the frame per side",
            "feature": "1/8 of the frame per side, so 1/64 of the pixels",
            "rgb": "the frame",
        },
        "mac_per_feature_px_released_decode": total,
        "mac_per_rgb_px_released_decode": total / 64.0,
        "share_of_released_decode": shares,
        "share_note": "denominator is the released decoder: upsample + 12 "
                      "trunk blocks + head, all on the feature grid. Seam "
                      "repair and the adapters are ours and are charged on top "
                      "of it, which is why the deepest exit exceeds 1.0.",
        "roles": roles,
        "router_v2": router,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  encoder side {enc:,}  decoder side {dec:,}  total {enc + dec:,}")
    for k, v in roles.items():
        print(f"  {k:<26} {v['params']:>10,} params  "
              f"{v['mac_per_out_px']:>10,} MAC/out-px")
    print(f"  router v2 {router['params']:,} params")
    for k, v in shares.items():
        print(f"  share {k:<24} {100 * v:8.4f}%")


if __name__ == "__main__":
    main()
