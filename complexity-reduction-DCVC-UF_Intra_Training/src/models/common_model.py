# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import torch.utils.checkpoint

from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..layers.layers import (LowerBound, QuantFunc, MSE_YUV_MEANS, mse_weighted_average,
                             get_mse_yuv_rgb)
from ..utils.common import generate_str


class CompressionModel(nn.Module):
    def __init__(self, z_channel):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.proxy = None
        self.bit_estimator_z = BitEstimator(self.qp_num(), z_channel)
        self.gaussian_encoder = GaussianEncoder()

        self.masks = {}

        self.prob_to_bits_factor = -1.0 / math.log(2.0)

        self.recon_range = 0.5

        # Training distortion, see set_mse_config(). Defaults are the official
        # 0.8 * YUV_geo(10:1:1) + 0.2 * RGB; None weights mean "use the
        # aggregation's own default triple".
        self.mse_rgb_weight = 0.2
        self.mse_yuv_mean = 'geometric'
        self.mse_yuv_weights = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def _reset_cuda_proxies(self):
        for module in self.modules():
            if hasattr(module, 'proxy'):
                try:
                    module.proxy = None
                except Exception:
                    pass

    def add_cdf_to_state_dict(self, state_dict):
        keys = ['quantized_cdf', 'cdf_length']
        state_dict.update({'gaussian_encoder.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.gaussian_encoder.get_cdf_info())})
        state_dict.update({'bit_estimator_z.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.bit_estimator_z.get_cdf_info())})
        return state_dict

    @staticmethod
    def add_noise(x):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5)
        return x + noise.detach()

    @staticmethod
    def get_loss_info(rd, loss, pixel_num, x=None):
        bpp_y = rd['bits_y'] / pixel_num
        bpp_z = rd['bits_z'] / pixel_num
        info = {
            'bpp_y': generate_str(bpp_y),
            'bpp_z': generate_str(bpp_z),
            'mse': generate_str(rd['mse']),
            'losses': generate_str(loss['losses']),
        }
        # Numeric batch-mean scalars for logging (e.g. wandb). Runs only when
        # get_loss_info=True, i.e. off the training-critical path, so it does
        # not affect optimization or numerics.
        scalars = {
            'loss': float(loss['loss'].item()),
            'bpp_y': float(bpp_y.mean().item()),
            'bpp_z': float(bpp_z.mean().item()),
            'bpp': float((bpp_y + bpp_z).mean().item()),
            'mse': float(rd['mse'].mean().item()),  # combined loss distortion
        }
        if x is not None:
            # Clean per-channel MSE on [0,1] YCbCr (MSE is shift-invariant, so
            # it is computed on the same -0.5-shifted tensors the model uses);
            # gives proper per-channel Y/Cb/Cr PSNR for logging.
            x_hat = rd['x_hat'].clamp(-0.5, 0.5)
            ch_mse = (x_hat - x).pow(2).mean(dim=(0, 2, 3))
            scalars['mse_y'] = float(ch_mse[0].item())
            scalars['mse_u'] = float(ch_mse[1].item())
            scalars['mse_v'] = float(ch_mse[2].item())
            scalars['mse_yuv'] = float(ch_mse.mean().item())
        info['scalars'] = scalars
        return info

    def set_mse_config(self, rgb_weight=None, yuv_mean=None, yuv_weights=None):
        """Select the training distortion. Defaults are the official
        0.8 * YUV_geo(10:1:1) + 0.2 * RGB; ``rgb_weight=0`` drops the RGB term,
        ``yuv_mean='arithmetic'`` swaps the geometric channel mean for the
        weighted arithmetic one (mlvc's image-model loss). Every argument left
        as None keeps the current value. This affects the loss only, not the
        architecture, so checkpoints stay interchangeable."""
        if rgb_weight is not None:
            if not 0.0 <= float(rgb_weight) < 1.0:
                raise ValueError(f'rgb_weight must be in [0, 1): {rgb_weight}')
            self.mse_rgb_weight = float(rgb_weight)
        if yuv_mean is not None:
            if yuv_mean not in MSE_YUV_MEANS:
                raise ValueError(f'unknown yuv_mean {yuv_mean!r}, '
                                 f'expected one of {sorted(MSE_YUV_MEANS)}')
            self.mse_yuv_mean = yuv_mean
            # A mean change invalidates weights that came from the other mean's
            # default, so fall back to this mean's default unless told otherwise.
            self.mse_yuv_weights = None
        if yuv_weights is not None:
            yuv_weights = tuple(float(w) for w in yuv_weights)
            if len(yuv_weights) != 3:
                raise ValueError(f'yuv_weights must have 3 entries: {yuv_weights}')
            if any(w < 0 for w in yuv_weights) or sum(yuv_weights) <= 0:
                raise ValueError(f'yuv_weights must be non-negative and not all '
                                 f'zero: {yuv_weights}')
            self.mse_yuv_weights = yuv_weights

    def get_mse(self, x, x_hat):
        # getattr: the TreeNet-4entropy model borrows this method without
        # inheriting CompressionModel.__init__.
        rgb_weight = getattr(self, 'mse_rgb_weight', 0.2)
        yuv_mean = getattr(self, 'mse_yuv_mean', 'geometric')
        yuv_weights = getattr(self, 'mse_yuv_weights', None)
        mse_yuv, mse_rgb = get_mse_yuv_rgb(x, x_hat, with_rgb=rgb_weight > 0)

        _, _, H, W = x.size()
        pixel_num = H * W
        mse = mse_weighted_average(mse_yuv, mse_rgb, pixel_num, rgb_weight=rgb_weight,
                                   yuv_mean=yuv_mean, yuv_weights=yuv_weights)
        return mse

    @staticmethod
    def get_one_mask(micro_mask, H, W):
        mask = torch.tensor(micro_mask, dtype=torch.bool)
        mask = mask.repeat((H + 1) // 2, (W + 1) // 2)
        mask = mask[None, None, :H, :W]
        return mask

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    def index_select_dim0(self, x, index):
        if x is None:
            return x
        out = torch.index_select(x, 0, index)[:, :, None, None]
        return out

    def probs_to_bits(self, probs):
        dtype = probs.dtype
        probs = probs.float()
        bits = torch.log(LowerBound.apply(probs, 1e-6)) * self.prob_to_bits_factor
        bits = LowerBound.apply(bits, 0)
        return bits.to(dtype)

    @staticmethod
    def process_with_mask(y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = QuantFunc.apply(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def qp_num():
        return 64

    def separate_prior_image(self, params):
        scales, means = params.chunk(2, 1)
        return scales, means

    def separate_prior_video(self, params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = LowerBound.apply(quant_step, 0.5)
        q_enc = 1. / quant_step
        q_dec = quant_step
        return q_enc, q_dec, scales, means

    def set_entropy_coder_parallel(self, entropy_coder_parallel):
        self.entropy_coder.set_entropy_coder_parallel(entropy_coder_parallel)

    def update(self, skip_thres):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, skip_thres=skip_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    @torch.no_grad()
    def get_mask_2x(self, B, C, H, W, device):
        curr_mask_str = f'{B}_{C}_{H}_{W}_2x'
        if curr_mask_str not in self.masks:
            assert C % 2 == 0
            m = torch.ones((B, C // 2, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 1)), H, W)
            m1 = self.get_one_mask(((0, 1), (1, 0)), H, W)

            mask_0 = torch.cat((m * m0, m * m1), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @torch.no_grad()
    def get_mask_4x(self, B, C, H, W, device):
        curr_mask_str = f'{B}_{C}_{H}_{W}_4x'
        if curr_mask_str not in self.masks:
            assert C % 4 == 0
            m = torch.ones((B, C // 4, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 0)), H, W)
            m1 = self.get_one_mask(((0, 1), (0, 0)), H, W)
            m2 = self.get_one_mask(((0, 0), (1, 0)), H, W)
            m3 = self.get_one_mask(((0, 0), (0, 1)), H, W)

            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_2 = mask_2.to(device=device)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            mask_3 = mask_3.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_y_bits(self, y, sigma):
        probs = self.gaussian_encoder.get_prob_train(y, sigma)
        return self.probs_to_bits(probs)

    def get_z_bits(self, z, index):
        probs = self.bit_estimator_z.get_prob(z, index)
        return self.probs_to_bits(probs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        res = super().load_state_dict(state_dict, *args, **kwargs)
        # CUDA proxy params are cached via .set_param(self.state_dict()) on first use.
        # When weights are reloaded, proxies must be reset to avoid using stale params.
        self._reset_cuda_proxies()
        return res

    def forward_prior_2x(self, y, common_params, y_spatial_prior):
        q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
        y = y * q_enc
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        means = y_spatial_prior(y_hat_0, common_params)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else y_res_0 + y_res_1
        y_q = None if y_q_0 is None else y_q_0 + y_q_1
        scales_hat = s_hat_0 + s_hat_1
        return y_res, y_q, y_hat, scales_hat

    def forward_prior_4x(self, y, q_enc, q_dec,
                         common_params, y_spatial_prior_reduction,
                         y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                         y_spatial_prior_adaptor_3, y_spatial_prior,
                         spatial_prior_has_scales=False):
        if q_enc is None:
            q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
            y = y * q_enc
        else:
            spatial_prior_has_scales = True
            scales, means = self.separate_prior_image(common_params)
            y = y * q_enc

        common_params = y_spatial_prior_reduction(common_params)
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_1(y_hat_so_far, common_params))
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_2(y_hat_so_far, common_params))
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_3(y_hat_so_far, common_params))
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = None if y_q_0 is None else (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        return y_res, y_q, y_hat, scales_hat
