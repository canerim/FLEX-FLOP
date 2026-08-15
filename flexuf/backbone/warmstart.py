"""Transfer stock DCVC-UF decoder weights into the multi-exit ladder.

Why a pure key remap and not a re-training
------------------------------------------
FLEX-FLOP tried training this architecture from scratch and every exit capped at
about 26 dB with no differentiation between exits at all (FORMAL_PAPER.md
section 3). Inheriting the released weights instead makes the deepest exit *be*
the teacher decoder from step 0, so training only ever has to learn the shallow
exits' corrections. That is the difference between a project that works and one
that does not.

The remap is exact — no tensor is reshaped, averaged, or interpolated. Every
target tensor either receives a source tensor byte-for-byte, or is a
zero-initialised adapter that did not exist in the source.

    dec_1.0.*            ->  upsample.*                (opening ResidualBlockUpsample)
    dec_1.{1..12}.*      ->  groups.{g}.{i}.*          with g = (n-1)//b, i = (n-1)%b
    dec_2.*              ->  head.*                    (the shared head)
    (nothing)            ->  adapters.{g}.*            new, zero-init => identity

Verification is not optional: `verify_bit_exact()` asserts max|stock - ladder|
is exactly 0.0. A control that cannot fail is not a control.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch

DCVC_ROOT = Path.home() / "DCVC"
if str(DCVC_ROOT) not in sys.path:
    sys.path.insert(0, str(DCVC_ROOT))

from ..config import N_TRUNK_BLOCKS  # noqa: E402

_DEC1 = re.compile(r"^dec_1\.(\d+)\.(.*)$")
_DEC2 = re.compile(r"^dec_2\.(.*)$")


def remap_decoder_state(
    src: Mapping[str, torch.Tensor], blocks_per_exit: int
) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    """Rewrite stock `IntraDecoder` keys into `MultiExitIntraDecoder` keys.

    Args:
        src: a state_dict of the stock IntraDecoder, or any dict whose keys have
            already been stripped down to `dec_1.*` / `dec_2.*`.
        blocks_per_exit: b = 12 / K.

    Returns:
        (remapped state dict, list of source keys that were not understood)
    """
    out: Dict[str, torch.Tensor] = {}
    unknown: list[str] = []

    for key, tensor in src.items():
        m1 = _DEC1.match(key)
        if m1:
            idx, rest = int(m1.group(1)), m1.group(2)
            if idx == 0:
                out[f"upsample.{rest}"] = tensor
            elif 1 <= idx <= N_TRUNK_BLOCKS:
                g, i = (idx - 1) // blocks_per_exit, (idx - 1) % blocks_per_exit
                out[f"groups.{g}.{i}.{rest}"] = tensor
            else:
                unknown.append(key)
            continue

        m2 = _DEC2.match(key)
        if m2:
            out[f"head.{m2.group(1)}"] = tensor
            continue

        unknown.append(key)

    return out, unknown


def extract_decoder_state(ckpt: Mapping) -> Dict[str, torch.Tensor]:
    """Pull the `dec.*` sub-tree out of a full DMCI checkpoint.

    Released UF checkpoints wrap the model state under a `state_dict` key and
    prefix decoder tensors with `dec.`. Both shapes are accepted so the same
    function works on a downloaded checkpoint and on `DMCI().state_dict()`.
    """
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
    picked = {
        k[len("dec.") :]: v for k, v in state.items() if k.startswith("dec.")
    }
    if not picked:  # already a bare IntraDecoder state_dict
        picked = {k: v for k, v in state.items() if k.startswith(("dec_1.", "dec_2."))}
    if not picked:
        raise KeyError(
            "no decoder tensors found: expected keys starting with 'dec.' or "
            f"'dec_1.'/'dec_2.', saw e.g. {list(state)[:5]}"
        )
    return picked


def load_into_ladder(model, ckpt: Mapping, *, strict_report: bool = True):
    """Warm-start `model` (a MultiExitIntraDecoder) from a UF checkpoint."""
    src = extract_decoder_state(ckpt)
    remapped, unknown = remap_decoder_state(src, model.cfg.blocks_per_exit)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Everything still missing must be one of the modules that did not exist in
    # stock UF — the exit adapters and the seam-repair block. Both are
    # zero-initialised, so a fresh model is exactly the released decoder until
    # they are trained. Anything else means the remap dropped a real tensor,
    # which would silently degrade the decoder.
    NEW_MODULES = ("adapters.", "seam_repair.")
    non_adapter_missing = [k for k in missing if not k.startswith(NEW_MODULES)]
    if strict_report:
        if non_adapter_missing:
            raise RuntimeError(
                f"warm-start dropped {len(non_adapter_missing)} non-adapter tensors, "
                f"e.g. {non_adapter_missing[:5]}"
            )
        if unexpected or unknown:
            raise RuntimeError(
                f"warm-start produced unmatched keys: unexpected={unexpected[:5]} "
                f"unknown_source={unknown[:5]}"
            )
    return {
        "transferred": len(remapped),
        "adapter_tensors_left_at_init": len(missing),
        "unknown_source_keys": unknown,
    }


@torch.no_grad()
def verify_bit_exact(ladder, stock, y_hat: torch.Tensor, quant_step: torch.Tensor) -> float:
    """Assert the deepest exit reproduces stock UF exactly. Returns max|diff|.

    This is the project's foundational control. If it is not 0.0, the ladder is
    not a re-expression of UF and every number measured against it is meaningless.
    """
    ladder.eval()
    stock.eval()
    ref = stock(y_hat, quant_step)
    got = ladder.forward_full(y_hat, quant_step, exit_idx=None)
    return (ref - got).abs().max().item()


def remap_ladder_to_stock(
    src: Mapping[str, torch.Tensor], blocks_per_exit: int
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`remap_decoder_state` — ladder keys back to stock UF keys.

    Needed by the evaluation control. Without it, loading a ladder state_dict into
    a stock `IntraDecoder` with `strict=False` silently matches *nothing* (the two
    use disjoint key names), leaving the stock model at its random init. The
    bit-exactness check would then compare a trained decoder against noise and
    report a huge difference, or — worse, if someone "fixed" it by loosening the
    tolerance — pass without ever having compared anything.

    Adapter tensors have no stock counterpart and are dropped: that is correct,
    because the control is precisely "the deepest exit takes no adapter".
    """
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        if key.startswith("adapters."):
            continue
        if key.startswith("upsample."):
            out[f"dec_1.0.{key[len('upsample.'):]}"] = tensor
        elif key.startswith("groups."):
            rest = key[len("groups."):]
            g_str, i_str, tail = rest.split(".", 2)
            idx = int(g_str) * blocks_per_exit + int(i_str) + 1
            out[f"dec_1.{idx}.{tail}"] = tensor
        elif key.startswith("head."):
            out[f"dec_2.{key[len('head.'):]}"] = tensor
    return out
