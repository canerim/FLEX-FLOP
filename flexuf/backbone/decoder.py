"""The multi-exit DCVC-UF intra decoder.

What this replaces
------------------
Stock UF decodes every latent pixel with the same 12 trunk blocks
(`~/DCVC/src/models/image_model.py:21-47`):

    y_hat -> ResidualBlockUpsample(256->384) -> 12x DepthConvBlock(384->384)
          -> *quant_step -> DepthConvBlock(384->192) -> PixelShuffle(8) -> RGB

Measured on a 1920x1088 frame (`scripts/mac_audit.py`, forward hooks on every
conv, MACs from real output shapes): 453.54 GMAC total, of which

    opening upsample     37.01 GMAC    8.16%   always runs
    12-block trunk      405.64 GMAC   89.44%   the part an exit can skip
    head                 10.89 GMAC    2.40%   always runs
    pointwise 1x1 convs 452.02 GMAC   99.66%   <- note this

Nearly nine tenths of the decode is a stack of twelve identical blocks that every
patch pays for regardless of how hard that patch is, and virtually all of it is
pointwise channel mixing. That is the opportunity, and it also dictates the
adapter design.

The structure, following ClassSR
--------------------------------
ClassSR decomposes an image into sub-images, decides how hard each one is, and
sends it through a branch whose capacity matches its difficulty. Here:

    shared stem (groups 0..j-1, full-frame, no seams)
        |
    patchify into 128x128-RGB tiles
        |
    per-tile ladder (groups j..K-1) -- each tile leaves at its own exit
        |
    per-exit 1x1 adapter -> shared head -> RGB

The exits are ClassSR's branches; the router (see `flexuf/router/`) is its
Class-Module. The difference is that our branches share weights with each other
by construction — a shallow exit is a prefix of a deep one — so there is no
3x parameter cost.

Why the adapter is a 1x1 convolution
------------------------------------
A DepthConvBlock costs 8C^2 + 9C MAC/px, of which the FFN alone is 6C^2 = 74.8%
and the only spatial operator is one 3x3 depthwise worth 9C = 0.3%. So exiting
early removes almost purely *pointwise channel-mixing* capacity. The matched
replacement is therefore also pointwise, and a 1x1 has a second property that
matters here: **zero receptive field, hence zero contribution to the patch
boundary penalty.** An adapter with a 3x3 in it would add seam damage exactly at
the tiles that took an early exit — the ones least able to afford it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DCVC_ROOT = Path.home() / "DCVC"
if str(DCVC_ROOT) not in sys.path:
    sys.path.insert(0, str(DCVC_ROOT))

from src.layers.layers import (  # noqa: E402
    DepthConvBlock,
    ResidualBlockUpsample,
    WSiLU,
    WSiLUChunkAdd,
)

from ..config import (  # noqa: E402
    LATENT_CH,
    UPSAMPLE_FACTOR,
    PRESHUFFLE_CH,
    SHUFFLE_FACTOR,
    TRUNK_CH,
    FlexUFConfig,
)


# ---------------------------------------------------------------------------
# Exit adapters
# ---------------------------------------------------------------------------
class Conv1x1Adapter(nn.Module):
    """A single trainable 1x1 convolution, residual, zero-initialised.

        Adapter(f) = f + W f,     W initialised to 0

    Zero-init makes the adapter exactly the identity before training. Two things
    follow: the deepest exit is bit-exact stock UF at step 0, and every shallow
    exit starts from "the decoder as it is" rather than from noise, so training
    only ever has to learn the correction.

    Cost: C^2 MAC/px = 1/8 of a DepthConvBlock's 8C^2, i.e. it buys back some of
    the skipped FFN capacity for an eighth of a block.
    """

    def __init__(self, channels: int = TRUNK_CH):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class FFNAdapter(nn.Module):
    """The pointwise expand/activate/contract pair the early exit skipped.

        Adapter(f) = f + PW_{C/r -> C}( WSiLUChunkAdd( PW_{C -> 4C/r}(f) ) )

    Same operator sequence as `DepthConvBlock.ffn` (`layers.py:146-150`), which
    is the 74.8% of each block an early exit gives up. Still entirely pointwise,
    so it too adds no patch-boundary penalty. More capacity than the plain 1x1
    at 2C^2/r MAC/px; experiment E4 measures whether that capacity is worth it.
    """

    def __init__(self, channels: int = TRUNK_CH, expand: int = 4):
        super().__init__()
        hidden = channels * expand
        self.pw_in = nn.Conv2d(channels, hidden, 1)
        self.act = WSiLUChunkAdd()  # 4:1 strided chunk-add -> hidden/4
        self.pw_out = nn.Conv2d(hidden // 4, channels, 1)
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw_out(self.act(self.pw_in(x)))


class SeamRepair(nn.Module):
    """A cheap full-frame pass that exists only to heal tile borders.

    Placed after unpatchify and before the head, where the canvas is whole again
    and the seams are visible as a grid — the error maps in results/samples show
    the deepest-exit error IS that grid and almost nothing else. Repairing it
    there costs one pass over the canvas instead of a multiplier on every
    per-tile block, which is what made the trunk halo unaffordable at 2.25x.

        Repair(f) = f + PW_{C->C}( WSiLU( DW3x3(f) ) ),   PW zero-initialised

    Zero-init keeps it exactly the identity until trained, so adding it cannot
    make anything worse and the bit-exactness control still passes.

    Cost, at C=384 and the feature grid being 1/64 of the RGB pixel count:

        depthwise 3x3   9C   =   3,456 MAC/feature-px  =  0.025% of the decode
        pointwise 1x1   C^2  = 147,456                 =  1.06%
        ------------------------------------------------------------------
        total                                            ~1.09%

    Against a per-tile trunk block at 7.45% of the decode, that is an eighth of
    one block spent full-frame. `depthwise_only` drops the pointwise for the
    0.025% version, which has spatial reach but no channel mixing — the seam is a
    spatial artefact, so that may be enough, and the two are separate config
    values precisely so the question gets measured rather than argued.
    """

    def __init__(self, channels: int = TRUNK_CH, depthwise_only: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                            padding_mode="replicate")
        self.act = WSiLU()
        self.pw = None if depthwise_only else nn.Conv2d(channels, channels, 1)
        last = self.dw if depthwise_only else self.pw
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dw(x))
        return x + (h if self.pw is None else self.pw(h))


def build_adapter(cfg: FlexUFConfig) -> nn.Module:
    if cfg.adapter_kind == "conv1x1":
        return Conv1x1Adapter(TRUNK_CH)
    return FFNAdapter(TRUNK_CH, cfg.adapter_expand)


# ---------------------------------------------------------------------------
# Patchify / unpatchify — pure reshapes, with an optional halo
# ---------------------------------------------------------------------------
def patchify(x: torch.Tensor, patch: int) -> tuple[torch.Tensor, int, int]:
    """[B,C,H,W] -> [B*nh*nw, C, patch, patch]. No conv, no arithmetic.

    This reshape is where the patch-independence penalty is born — once tiles are
    separate batch elements, every later 3x3 sees zero padding where a neighbour
    should be — and it is also where every FLOP of saving comes from, because
    only now can individual tiles stop early.
    """
    b, c, h, w = x.shape
    assert h % patch == 0 and w % patch == 0, (
        f"feature map {h}x{w} is not divisible by patch {patch}"
    )
    nh, nw = h // patch, w // patch
    out = (
        x.view(b, c, nh, patch, nw, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b * nh * nw, c, patch, patch)
    )
    return out, nh, nw


def unpatchify(x: torch.Tensor, nh: int, nw: int, batch: int = 1) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    _, c, ph, pw = x.shape
    return (
        x.view(batch, nh, nw, c, ph, pw)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, c, nh * ph, nw * pw)
    )


def patchify_with_halo(
    x: torch.Tensor, patch: int, halo: int
) -> tuple[torch.Tensor, int, int]:
    """Tiles of size `patch + 2*halo`, each carrying real neighbour context.

    The halo is what lets a tile be decoded independently without its border
    being convolved against zeros. It is sized in `flexuf.config` from the actual
    receptive field of the per-patch region: (K-j)*b trunk blocks, each with one
    3x3 depthwise, reach (K-j)*b feature pixels past the border.

    Replicate padding at the frame edge, real features everywhere else.
    """
    b, c, h, w = x.shape
    assert h % patch == 0 and w % patch == 0
    nh, nw = h // patch, w // patch
    padded = F.pad(x, (halo,) * 4, mode="replicate")
    side = patch + 2 * halo
    # unfold gives every (patch+2*halo) window at stride `patch`
    tiles = padded.unfold(2, side, patch).unfold(3, side, patch)
    # [B, C, nh, nw, side, side] -> [B*nh*nw, C, side, side]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(b * nh * nw, c, side, side)
    return tiles.contiguous(), nh, nw


def crop_halo(x: torch.Tensor, halo: int, scale: int = 1) -> torch.Tensor:
    """Drop the halo band after processing. `scale` accounts for upsampling."""
    h = halo * scale
    return x[:, :, h:-h, h:-h] if h > 0 else x


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------
class MultiExitIntraDecoder(nn.Module):
    """DCVC-UF's IntraDecoder, re-expressed as a K-exit ladder.

    Module names are chosen so stock UF weights transfer by a pure key remap
    (`warmstart.py`):

        upsample      <- dec_1[0]            (ResidualBlockUpsample 256->384)
        groups[g][i]  <- dec_1[1 + g*b + i]  (DepthConvBlock 384->384)
        adapters[g]    new, zero-init, g < K-1
        head          <- dec_2               (DepthConvBlock 384->192)
    """

    def __init__(self, cfg: Optional[FlexUFConfig] = None):
        super().__init__()
        self.cfg = cfg or FlexUFConfig()
        K, b = self.cfg.num_exits, self.cfg.blocks_per_exit

        self.upsample = ResidualBlockUpsample(LATENT_CH, TRUNK_CH)
        self.groups = nn.ModuleList(
            nn.Sequential(*[DepthConvBlock(TRUNK_CH, TRUNK_CH) for _ in range(b)])
            for _ in range(K)
        )
        # K-1 adapters: the deepest exit takes the raw feature, which is what
        # makes it bit-exact stock UF.
        self.adapters = nn.ModuleList(build_adapter(self.cfg) for _ in range(K - 1))
        self.head = DepthConvBlock(TRUNK_CH, PRESHUFFLE_CH)
        self.seam_repair = (
            SeamRepair(TRUNK_CH, self.cfg.seam_repair == "depthwise")
            if self.cfg.seam_repair != "none" else None
        )

    # -- tile-border padding -------------------------------------------------
    def _set_tile_padding(self, mode: str, first_group: int):
        """Switch the depthwise convs of groups >= first_group to `mode`.

        Only 3x3 depthwise convolutions have any spatial extent, so they are the
        only place a tile border can meet padding at all. Flipping the mode costs
        nothing — it is a string on the module, consumed by the same kernel.

        Applied only to the groups that run per-tile, and restored afterwards, so
        full-frame decode is untouched and `forward_full` stays bit-exact against
        stock UF.
        """
        for g in range(first_group, len(self.groups)):
            for m in self.groups[g].modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.groups > 1:
                    m.padding_mode = mode
                    m._reversed_padding_repeated_twice = [1, 1, 1, 1]

    # -- pieces --------------------------------------------------------------
    def _at_exit(self, feat: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """Feature handed to the head from exit `exit_idx`."""
        if exit_idx >= self.cfg.num_exits - 1:
            return feat  # raw — the bit-exact path
        return self.adapters[exit_idx](feat)

    def _apply_head(self, feat: torch.Tensor, quant_step: torch.Tensor) -> torch.Tensor:
        """quant_step scaling, head block, PixelShuffle(8).

        Mirrors stock UF exactly (`image_model.py:42-47`): the per-QP scale is
        applied to the trunk output *before* the head.
        """
        return F.pixel_shuffle(self.head(feat * quant_step), SHUFFLE_FACTOR)

    # -- uniform-depth decode ------------------------------------------------
    def forward_full(
        self,
        y_hat: torch.Tensor,
        quant_step: torch.Tensor,
        exit_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode at one uniform depth, no patching.

        `exit_idx=None` is the anchor and must reproduce stock UF bit-exactly.
        A fixed `exit_idx` gives the "every tile at depth k" reference curve, and
        is what the Eq.(6)-(7) joint objective supervises during training.
        """
        stop = self.cfg.num_exits - 1 if exit_idx is None else exit_idx
        feat = self.upsample(y_hat)
        for g in range(stop + 1):
            feat = self.groups[g](feat)
        return self._apply_head(self._at_exit(feat, stop), quant_step)

    def forward_all_exits(
        self, y_hat: torch.Tensor, quant_step: torch.Tensor
    ) -> list[torch.Tensor]:
        """Every exit's reconstruction from ONE shared trunk pass.

        This is the efficient form of the Eq.(6)-(7) joint objective: the trunk
        runs once and each exit taps the running feature, instead of K separate
        forward passes. Cost is one full trunk plus K heads.
        """
        feat = self.upsample(y_hat)
        outs = []
        for g in range(self.cfg.num_exits):
            feat = self.groups[g](feat)
            outs.append(self._apply_head(self._at_exit(feat, g), quant_step))
        return outs

    # -- the deployed hybrid decode -----------------------------------------
    def forward(
        self,
        y_hat: torch.Tensor,
        quant_step: torch.Tensor,
        exit_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Depth-adaptive decode: shared full-frame stem, per-tile suffix.

        Args:
            y_hat: [1, 256, h, w] latent, entropy-decoded once, full-frame.
            quant_step: [1, 384, 1, 1] per-QP scale.
            exit_map: [P] int64 exit per tile, P = (2h/Fp)*(2w/Fp). None means
                "everyone deepest", which at j=K must equal `forward_full()`.

        Order of operations:
          1. upsample + groups 0..j-1   full-frame  (no seams, no saving)
          2. patchify with halo         reshape     (seams born, 0 FLOP)
          3. groups j..K-1              per-tile    (tiles drop out at their exit)
          4. 1x1 adapter                per-tile    (no receptive field, no seam)
          5. crop halo, unpatchify      reshape
          6. head                       full-frame  (heals the remaining seam)
        """
        # Catch the broadcast trap at the door rather than in each caller.
        #
        # quant_step carries one row per image. Handing a batch-B quant_step to a
        # single-image decode makes [1,C,H,W] * [B,C,1,1] broadcast the result
        # silently back up to batch B — no error, just a wrong-shaped answer that
        # surfaces somewhere unrelated. It has now happened twice, in model.py
        # and in rd_curve.py, because nothing tied the call sites together.
        if quant_step.dim() == 4 and quant_step.shape[0] not in (1, y_hat.shape[0]):
            raise ValueError(
                f"quant_step batch {quant_step.shape[0]} does not match latent "
                f"batch {y_hat.shape[0]}; slice it per image "
                f"(quant_step[i:i+1]) when decoding one image at a time"
            )

        cfg = self.cfg
        # The trunk halo is its own knob and defaults to ZERO. Carrying the
        # router's halo through the per-tile trunk multiplies that work by
        # ((F+2h)/F)^2 = 2.25x at the default geometry, which measured 1.745x the
        # cost of a plain full decode even with every tile at the deepest exit.
        # The seam is handled by the full-frame head instead.
        K, j, Fp = cfg.num_exits, cfg.split_depth, cfg.feature_patch
        halo = cfg.trunk_halo * UPSAMPLE_FACTOR

        # ---- 1. shared stem ------------------------------------------------
        feat = self.upsample(y_hat)
        for g in range(j):
            feat = self.groups[g](feat)

        if j >= K:  # the correctness control: nothing runs per-tile
            return self._apply_head(self._at_exit(feat, K - 1), quant_step)

        # ---- 2. patchify ---------------------------------------------------
        if halo > 0:
            tiles, nh, nw = patchify_with_halo(feat, Fp, halo)
        else:
            tiles, nh, nw = patchify(feat, Fp)
        n_tiles = tiles.shape[0]

        if exit_map is None:
            exit_map = torch.full((n_tiles,), K - 1, dtype=torch.long, device=tiles.device)
        exit_map = exit_map.to(tiles.device).long().clamp(min=j, max=K - 1)
        assert exit_map.shape == (n_tiles,), (
            f"exit_map has {tuple(exit_map.shape)}, expected ({n_tiles},)"
        )

        canvas = torch.zeros_like(tiles)

        # Tile borders meet padding from here on. Zeros is the stock behaviour
        # and a poor estimate of the missing neighbour; replicating the edge
        # removes 80% of the seam at qp63 for no compute. Restored below so
        # full-frame decode is unaffected.
        if cfg.tile_pad_mode != "zeros":
            self._set_tile_padding(cfg.tile_pad_mode, j)

        # ---- 3+4. per-tile suffix -----------------------------------------
        # `active` is the shrinking set of tiles still climbing the ladder.
        # Every group runs on fewer tiles than the last — that shrinkage IS the
        # saving.
        active = torch.arange(n_tiles, device=tiles.device)
        work = tiles
        for g in range(j, K):
            work = self.groups[g](work)
            leaving = exit_map[active] == g
            if leaving.any():
                canvas[active[leaving]] = self._at_exit(work[leaving], g)
                keep = ~leaving
                if not keep.any():
                    break
                work = work[keep]
                active = active[keep]

        if cfg.tile_pad_mode != "zeros":
            self._set_tile_padding("zeros", j)

        # ---- 5. drop halo, stitch ------------------------------------------
        stitched = unpatchify(crop_halo(canvas, halo) if halo > 0 else canvas,
                              nh, nw, batch=feat.shape[0])

        # ---- 5b. heal the seams, full-frame ---------------------------------
        # Only meaningful here: the canvas is whole, so a 3x3 finally sees across
        # a tile boundary instead of into padding.
        if self.seam_repair is not None:
            stitched = self.seam_repair(stitched)

        # ---- 6. head -------------------------------------------------------
        if cfg.full_frame_head:
            return self._apply_head(stitched, quant_step)
        return self._head_per_tile(stitched, quant_step, nh, nw)

    # -- per-tile head (the ablation arm) ------------------------------------
    def _head_per_tile(
        self, stitched: torch.Tensor, quant_step: torch.Tensor, nh: int, nw: int
    ) -> torch.Tensor:
        """Head applied tile-by-tile with a halo, instead of over the whole canvas.

        Exists to *measure* the head's contribution to the seam rather than
        assume it. With halo=0 this is the naive per-patch decode; with the
        configured halo it should land within a few hundredths of a dB of the
        full-frame arm.
        """
        cfg = self.cfg
        halo, Fp = cfg.feature_halo, cfg.feature_patch
        tiles, _, _ = patchify_with_halo(stitched, Fp, halo)
        rgb = self._apply_head(tiles, quant_step)
        rgb = crop_halo(rgb, halo, scale=SHUFFLE_FACTOR)
        return unpatchify(rgb, nh, nw, batch=stitched.shape[0])

    # -- bookkeeping ---------------------------------------------------------
    def adapter_parameters(self):
        return self.adapters.parameters()

    def freeze_backbone(self):
        """Freeze everything inherited from UF; leave adapters trainable.

        Only meaningful in the warm-start regime. In the from-scratch regime the
        whole model trains jointly under Eq. (6).
        """
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("adapters.")
        return self
