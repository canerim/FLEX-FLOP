"""The multi-exit DCVC-UF intra codec.

`FlexUFIntra` is stock `DMCI` with its decoder swapped for the K-exit ladder.
Everything on the encode side — analysis transform, hyperprior, the 4-step
spatial-prior entropy model, the 64 QP scale tables — is byte-for-byte the
Microsoft model, because the point of the project is to change *where decode
compute is spent*, not to change the codec.

That has one consequence worth stating plainly, because it shapes the whole
objective: **the rate does not depend on the exit.** Entropy decoding happens
once, full-frame, before the decoder is ever called. Every exit reads the same
y_hat and therefore costs the same bits. So the RD trade-off per exit is

    L_k = lambda * MSE_k + BPP        with BPP identical across k

and the only thing an exit changes is distortion and compute. This is why the
frontier is drawn as (compute saved) vs (dB lost) at fixed rate, and why the
router's job is purely "how much decode does this tile need", never "how many
bits".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

DCVC_ROOT = Path.home() / "DCVC"
if str(DCVC_ROOT) not in sys.path:
    sys.path.insert(0, str(DCVC_ROOT))

from src.models.image_model import DMCI  # noqa: E402

from .backbone.decoder import MultiExitIntraDecoder  # noqa: E402
from .config import FlexUFConfig  # noqa: E402


class FlexUFIntra(DMCI):
    """DMCI with a multi-exit decoder.

    Adds three forward modes on top of stock DMCI:

    - `forward_all_exits`  — every exit's reconstruction from one trunk pass.
      This is what the Eq. (6)-(7) joint objective consumes during training.
    - `forward_routed`     — the deployed decode: shared stem, per-tile suffix,
      each tile at the exit the router picked.
    - stock `forward`      — unchanged, so the model is still a drop-in DMCI and
      the deepest exit remains directly comparable to Microsoft's numbers.
    """

    def __init__(self, cfg: Optional[FlexUFConfig] = None):
        super().__init__()
        self.cfg = cfg or FlexUFConfig()
        # Replace the decoder. Constructed after super().__init__() so it does
        # not get caught by DMCI's _initialize_weights(), which would overwrite
        # the deliberate zero-init on the adapters.
        self.dec = MultiExitIntraDecoder(self.cfg)
        self._zero_init_adapters()

    def _zero_init_adapters(self):
        """Re-assert the identity guarantee after any weight init pass."""
        for adapter in self.dec.adapters:
            last = adapter.conv if hasattr(adapter, "conv") else adapter.pw_out
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    # -- the shared front end -----------------------------------------------
    def _encode_to_latent(self, x: torch.Tensor, qp):
        """Everything up to y_hat: analysis, hyperprior, spatial-prior entropy model.

        Lifted verbatim from `DMCI.forward_one_frame` (`image_model.py:150-168`)
        so the encode path stays identical to Microsoft's.
        """
        curr_q_enc = self.index_select_dim0(self.q_scale_enc, qp)
        curr_q_dec = self.index_select_dim0(self.q_scale_dec, qp)
        curr_y_q_enc = self.index_select_dim0(self.q_scale_y_enc, qp)
        curr_y_q_dec = self.index_select_dim0(self.q_scale_y_dec, qp)

        y = self.enc(x, curr_q_enc)
        z = self.hyper_enc(y)
        z_hat = self.quant(z) if hasattr(self, "quant") else _quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW]
        y_res, y_q, y_hat, scales_hat = self.forward_prior_4x(
            y,
            curr_y_q_enc,
            curr_y_q_dec,
            params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
        )
        return y_hat, curr_q_dec, dict(y_res=y_res, z=z, scales_hat=scales_hat)

    def _rate(self, aux, qp, pixel_num):
        """Bits per pixel. Independent of the exit — see the module docstring."""
        bits_y = self.get_y_bits(self.add_noise(aux["y_res"]), aux["scales_hat"])
        bits_z = self.get_z_bits(self.add_noise(aux["z"]), qp)
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        return (bits_y + bits_z) / pixel_num, bits_y, bits_z

    # -- training forward ----------------------------------------------------
    def forward_all_exits(self, x: torch.Tensor, qp):
        """Reconstructions at every exit, plus the shared rate.

        Returns a dict with `x_hats` (list of K tensors), `mses` (list of K),
        and the single shared `bpp`.
        """
        _, _, H, W = x.size()
        pixel_num = H * W
        y_hat, curr_q_dec, aux = self._encode_to_latent(x, qp)

        x_hats = self.dec.forward_all_exits(y_hat, curr_q_dec)
        mses = [self.get_mse(x, xh) for xh in x_hats]
        bpp, bits_y, bits_z = self._rate(aux, qp, pixel_num)

        return {
            "x_hats": x_hats,
            "mses": mses,
            "bpp": bpp,
            "bits_y": bits_y,
            "bits_z": bits_z,
        }

    # -- deployed forward ----------------------------------------------------
    @torch.no_grad()
    def forward_routed(self, x: torch.Tensor, qp, exit_map: torch.Tensor):
        """The decode as it would actually run: per-tile depth from `exit_map`."""
        _, _, H, W = x.size()
        y_hat, curr_q_dec, aux = self._encode_to_latent(x, qp)
        x_hat = self.dec(y_hat, curr_q_dec, exit_map=exit_map)
        bpp, _, _ = self._rate(aux, qp, H * W)
        return {"x_hat": x_hat, "mse": self.get_mse(x, x_hat), "bpp": bpp}

    @torch.no_grad()
    def latent_of(self, x: torch.Tensor, qp):
        """y_hat and the decoder quant step — what the router reads."""
        y_hat, curr_q_dec, _ = self._encode_to_latent(x, qp)
        return y_hat, curr_q_dec


def _quant(z):
    from src.layers.layers import QuantFunc

    return QuantFunc.apply(z)
