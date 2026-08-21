# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Function

from ..utils.transforms import ycbcr2rgb


def bit_estimator_z_prob(x, h, b, a):
    # this is accumulated prob
    for i in range(4):
        x = x * F.softplus(h[:, :, i:i+1, None]) + b[:, :, i:i+1, None]
        if i != 3:
            x = x + torch.tanh(x) * torch.tanh(a[:, :, i:i+1, None])
    return torch.sigmoid(x)


def bit_estimator_z_fwd(x, h, b, a):
    dtype = x.dtype
    x = x.float()
    h = h.float()
    b = b.float()
    a = a.float()
    lower = bit_estimator_z_prob(x - 0.5, h, b, a)
    upper = bit_estimator_z_prob(x + 0.5, h, b, a)
    prob = upper - lower
    return prob.to(dtype)


def get_mse_yuv_rgb(x, x_hat, with_rgb=True):
    mse_yuv = torch.sum(F.mse_loss(x, x_hat, reduction='none'), dim=(2, 3))
    if not with_rgb:
        # YUV-only loss: skip the colour conversion entirely instead of
        # computing an RGB term that gets multiplied by zero.
        return mse_yuv, None
    org_rgb = ycbcr2rgb(x, clamp=False)
    rec_rgb = ycbcr2rgb(x_hat, clamp=False)
    mse_rgb = torch.sum(F.mse_loss(org_rgb, rec_rgb, reduction='none'), dim=(1, 2, 3))
    return mse_yuv, mse_rgb


# Per-channel (Y, U, V) weights used when the caller does not pass any.
# The geometric default is DCVC-UF's official 10:1:1; the arithmetic default
# follows mlvc's mse_y_weight=4, so switching aggregation reproduces the loss
# mlvc's image model (loss_total_rdc_mse) actually trains with.
MSE_YUV_WEIGHTS_GEOMETRIC = (10.0, 1.0, 1.0)
MSE_YUV_WEIGHTS_ARITHMETIC = (4.0, 1.0, 1.0)


def _mse_yuv_geometric(mse_y, mse_u, mse_v, weights):
    """Weighted geometric mean, i.e. a fixed weighted average of the per-channel
    PSNRs. Each channel's share of the gradient is its weight, independent of
    how large that channel's error currently is."""
    w_y, w_u, w_v = weights
    scale = 1.0 / (w_y + w_u + w_v)
    return 3 * torch.exp(
        scale * (w_y * torch.log(torch.clamp_min(mse_y, 1e-6))
                 + w_u * torch.log(torch.clamp_min(mse_u, 1e-6))
                 + w_v * torch.log(torch.clamp_min(mse_v, 1e-6))))


def _mse_yuv_arithmetic(mse_y, mse_u, mse_v, weights):
    """Weighted arithmetic mean. The exchange rate between absolute squared
    errors is fixed, so whichever channel currently has the largest error
    dominates the gradient."""
    w_y, w_u, w_v = weights
    scale = 3.0 / (w_y + w_u + w_v)
    return scale * (w_y * mse_y + w_u * mse_u + w_v * mse_v)


# Both aggregations use the same normalisation: three channels with equal MSE m
# map to 3m, so switching between them does not rescale the distortion term and
# the lambdas stay comparable.
MSE_YUV_MEANS = {
    'geometric': (_mse_yuv_geometric, MSE_YUV_WEIGHTS_GEOMETRIC),
    'arithmetic': (_mse_yuv_arithmetic, MSE_YUV_WEIGHTS_ARITHMETIC),
}


def mse_weighted_average(mse_yuv, mse_rgb, pixel_num, rgb_weight=0.2,
                         yuv_mean='geometric', yuv_weights=None):
    """Combine the YUV and RGB distortion terms.

    ``yuv_mean`` selects how the three channels are aggregated ('geometric' or
    'arithmetic'), ``yuv_weights`` is the (Y, U, V) weight triple and defaults
    to the aggregation's own default. ``rgb_weight`` is the weight of the RGB
    term; the YUV term takes the remaining ``1 - rgb_weight``, and at 0 the
    ``mse_rgb`` argument may be None.

    The defaults reproduce the official 0.8 * YUV_geo(10:1:1) + 0.2 * RGB loss.
    """
    if yuv_mean not in MSE_YUV_MEANS:
        raise ValueError(f'unknown yuv_mean {yuv_mean!r}, '
                         f'expected one of {sorted(MSE_YUV_MEANS)}')
    aggregate, default_weights = MSE_YUV_MEANS[yuv_mean]
    if yuv_weights is None:
        yuv_weights = default_weights

    dtype = mse_yuv.dtype
    mse_yuv = mse_yuv.float()
    mse_yuv = mse_yuv / pixel_num
    mse_yuv = aggregate(mse_yuv[:, 0], mse_yuv[:, 1], mse_yuv[:, 2], yuv_weights)
    if rgb_weight <= 0:
        return mse_yuv.to(dtype)
    mse_rgb = mse_rgb.float()
    mse_rgb = mse_rgb / pixel_num
    mse = mse_yuv * (1.0 - rgb_weight) + mse_rgb * rgb_weight
    return mse.to(dtype)


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs)
        ctx.bound = bound
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad):
        inputs = ctx.saved_tensors[0]
        bound = ctx.bound

        pass_through_1 = inputs >= bound
        pass_through_2 = grad < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through * grad, None


class QuantFunc(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, force_bias=False):
        super().__init__()
        has_bias = (kernel_size > 1) or force_bias
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, bias=has_bias),
            nn.PixelShuffle(2),
        )
        self.padding = padding

    def forward(self, x):
        return self.conv(x)


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsilu = WSiLU()

    def forward(self, x):
        x = self.wsilu(x)
        x1 = x[:, 0::4, :, :]
        x2 = x[:, 1::4, :, :]
        x3 = x[:, 2::4, :, :]
        x4 = x[:, 3::4, :, :]
        return x1 + x2 + x3 + x4


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        ch_ratio = 1
        if dcb2:
            assert not shortcut
            ch_ratio = 2
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // ch_ratio, 1),
            WSiLU(),
            nn.Conv2d(out_ch // ch_ratio, out_ch // ch_ratio, 3, padding=1,
                      groups=out_ch // ch_ratio),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4 // ch_ratio, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True, force_bias=False):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1, force_bias=force_bias)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True):
        super().__init__()
        self.down = nn.Conv2d(in_ch * 4, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = F.pixel_unshuffle(x, 2)
        out = self.down(out)
        out = self.conv(out)
        return out
