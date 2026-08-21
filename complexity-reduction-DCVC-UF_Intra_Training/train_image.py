# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import contextlib
import hashlib
import json
import logging
import math
import numpy as np
import os
import random
import sys
import tempfile
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.image_dataset import ImageFolder, ImageValFolder
from src.layers.layers import MSE_YUV_MEANS
from src.models.image_model import DMCI
from src.utils.common import start_train, save_ckpt, save_status, get_current_device, \
    get_training_lambdas, get_dataloader, init_train, cleanup_train, load_existing_weights, \
    wrap_ddp, get_state_dict


LOGGER = logging.getLogger(__name__)
TRAIN_METRIC_MONITOR_INTERVAL = 200

def get_training_strategy():
    # epoch is for referencing purpose
    # lr is the learning rate for the current epoch
    training_strategy = \
        [[0,   2e-4, 256, 256]] * 45 + \
        [[49,  5e-5, 256, 256]] * 25 + \
        [[69,  1e-5, 256, 256]] * 20 + \
        [[90,  2e-4, 512, 512]] * 5 + \
        [[95,  5e-5, 512, 512]] * 4 + \
        [[99,  1e-5, 512, 512]] * 4 + \
        [[103, 1e-6, 512, 512]] * 2 + \
        [[105, 1e-6, 512, 512]]  # noqa: E501 E221
        # epo, lr,   patch_size  # noqa: E116 E501

    return training_strategy


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', default=105, type=int)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('--mse_rgb_weight', type=float, default=0.2,
                        help='weight of the RGB term in the training distortion; the YUV '
                             'term takes the rest. 0.2 is the official '
                             '0.8*YUV + 0.2*RGB loss, 0 trains on YUV only. Note that at '
                             'weight 0 the YUV term is no longer scaled by 0.8, so the '
                             'effective lambda grows by 1.25x; pass lambdas scaled by 0.8 '
                             'to land on the same rate point')
    parser.add_argument('--mse_yuv_mean', type=str, default='geometric',
                        choices=sorted(MSE_YUV_MEANS),
                        help='how the three YUV channels are aggregated. "geometric" is '
                             'the official DCVC-UF loss (a fixed weighted average of the '
                             'per-channel PSNRs, so each channel keeps its share of the '
                             'gradient no matter how bad it currently is); "arithmetic" '
                             'is the weighted sum of squared errors that mlvc\'s image '
                             'model trains with, under which a degraded channel '
                             'automatically pulls harder. Both are normalised the same '
                             'way, so the lambdas stay comparable')
    parser.add_argument('--mse_yuv_weights', type=float, nargs=3,
                        metavar=('Y', 'U', 'V'), default=None,
                        help='per-channel weights for --mse_yuv_mean. Default follows the '
                             'aggregation: 10 1 1 for geometric (official DCVC-UF), '
                             '4 1 1 for arithmetic (mlvc default). Only the ratio matters, '
                             'the triple is renormalised internally')
    parser.add_argument('-n', '--num_workers', type=int, default=8,
                        help='Dataloaders worker per trainer')
    parser.add_argument('--amp', action='store_true',
                        help='run the training forward/backward under bf16 autocast. '
                             'Weights and optimizer state stay fp32, and the entropy model '
                             'already forces fp32 internally, so checkpoints remain '
                             'interchangeable with fp32 runs. Validation always runs in fp32')
    parser.add_argument('--compile', nargs='?', const='full', default='off',
                        choices=['off', 'full', 'parts'],
                        help='torch.compile the training graph (Triton required). "full" '
                             'compiles the whole model; "parts" compiles only the enc/dec '
                             'transforms, which is the fallback for models whose hyper/prior '
                             'backward kernels crash Triton (the official DMCI under bf16 on '
                             'sm_120). Checkpoints are unaffected either way')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode, used together with --compile. '
                             '"reduce-overhead" adds CUDA graphs, which removes most of the '
                             'per-step launch overhead; it needs the shapes to stay constant '
                             'within a phase, which the training schedule guarantees')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for model init, sampling order, augmentation and the '
                             'entropy-model noise. Default None keeps the previous behaviour '
                             '(a different stream every run). A fresh run with a fixed seed '
                             'is reproducible; after a resume the model/optimizer state is '
                             'exact but the augmentation stream restarts')
    parser.add_argument('--deterministic', action='store_true',
                        help='on top of --seed, force bit-exact reruns: disables the cuDNN '
                             'autotuner and selects deterministic kernels. Costs throughput, '
                             'so use it for debugging or exact ablations, not production runs')
    parser.add_argument('--fused_adam', action='store_true',
                        help='use the fused AdamW kernel instead of the default foreach '
                             'implementation (CUDA only); collapses the per-parameter update '
                             'into a single kernel launch')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models')
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--train_qp', type=int, default=None,
                        help='pin training to a single QP index (e.g. 63 for single-rate '
                             'training); default None samples all 64 QPs randomly')

    # ---- checkpoint retention ----
    parser.add_argument('--keep_status_num', type=int, default=2,
                        help='how many rolling status_epo*.pth.tar resume points to keep '
                             '(weights + optimizer state). 2 keeps a fallback in case the '
                             'newest one is unreadable')
    parser.add_argument('--archive_interval', type=int, default=0,
                        help='additionally archive a weights-only snapshot every N epochs '
                             'to <save_dir>/archives/ckpt_epoNNN.pth.tar (and on the final '
                             'epoch); 0 disables archiving')

    # ---- wandb monitoring (optional; training runs fine without wandb) ----
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='wandb logging mode; "disabled" turns monitoring off')
    parser.add_argument('--wandb_project', type=str,
                        default=os.environ.get('WANDB_PROJECT', 'DCVC-UF'))
    parser.add_argument('--wandb_entity', type=str,
                        default=os.environ.get('WANDB_ENTITY', None))
    parser.add_argument('--exp_name', type=str, default=None,
                        help='wandb run name; defaults to the save_dir basename')
    parser.add_argument('--wandb_step_interval', type=int, default=100,
                        help='log a current-batch train-step/* snapshot every N successful '
                             'optimizer steps; epoch-level train-* metrics are logged separately')
    parser.add_argument('--ddp_timeout_minutes', type=int, default=360,
                        help='DDP collective timeout; should exceed the longest rank-0 '
                             'reference/validation pass')

    # ---- validation (optional; skipped entirely when --val_dataset is unset) ----
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='held-out image root for validation; if unset, no validation')
    parser.add_argument('--val_list', type=str, default=None,
                        help='val list file (.json or .txt); default: description.json in val root')
    parser.add_argument('--val_qps', type=str, default='0,15,31,47,63',
                        help='comma-separated QP indices to evaluate (RD curve points)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='run validation every N epochs (and on the final epoch)')
    parser.add_argument('--val_max_side', type=int, default=1280,
                        help='center-crop val images to this max side (64-aligned) to bound '
                             'memory/time; 0 disables cropping')
    parser.add_argument('--ref_ckpt', type=str, default=None,
                        help='official pretrained DMCI checkpoint; its RD curve is drawn as a '
                             'reference on the validation plots (evaluated once on the val set)')
    parser.add_argument('--ref_qps', type=str, default='0,7,15,23,31,39,47,55,63',
                        help='QP indices at which to evaluate the reference model (the '
                             'reference RD curve spans these points regardless of --val_qps)')
    parser.add_argument('--ref_cache', type=str, default=None,
                        help='where the reference RD points are cached; default '
                             '<save_dir>/ref_rd_points.json. Point several runs at one '
                             'shared file so the official model is evaluated only once '
                             'instead of once per save_dir')

    args = parser.parse_args(argv)
    return args


def _parse_qp_values(raw_value, option_name):
    try:
        values = [int(value.strip()) for value in raw_value.split(',') if value.strip()]
    except ValueError as e:
        raise ValueError(f'{option_name} must be a comma-separated integer list') from e
    if not values:
        raise ValueError(f'{option_name} must contain at least one QP')
    if len(set(values)) != len(values):
        raise ValueError(f'{option_name} contains duplicate QPs: {values}')
    invalid = [value for value in values if not 0 <= value < DMCI.qp_num()]
    if invalid:
        raise ValueError(
            f'{option_name} values must be in [0, {DMCI.qp_num() - 1}]: {invalid}'
        )
    return sorted(values)


def validate_args(args):
    if args.train_qp is not None and not 0 <= args.train_qp < DMCI.qp_num():
        raise ValueError(f'--train_qp must be in [0, {DMCI.qp_num() - 1}]')
    if not 0.0 <= args.mse_rgb_weight < 1.0:
        raise ValueError('--mse_rgb_weight must be in [0, 1)')
    if args.mse_yuv_weights is not None:
        if any(w < 0 for w in args.mse_yuv_weights) or sum(args.mse_yuv_weights) <= 0:
            raise ValueError('--mse_yuv_weights must be non-negative and not all zero')
    # Resolve the effective triple so the log line and the run config record what
    # actually trained, not "None".
    args.mse_yuv_weight_values = tuple(
        args.mse_yuv_weights if args.mse_yuv_weights is not None
        else MSE_YUV_MEANS[args.mse_yuv_mean][1]
    )
    if args.val_interval <= 0:
        raise ValueError('--val_interval must be positive')
    if args.val_max_side != 0 and args.val_max_side < 64:
        raise ValueError('--val_max_side must be 0 (disabled) or at least 64')
    if args.wandb_step_interval < 0:
        raise ValueError('--wandb_step_interval must be non-negative')
    if args.keep_status_num < 1:
        raise ValueError('--keep_status_num must be at least 1')
    if args.archive_interval < 0:
        raise ValueError('--archive_interval must be non-negative')
    if args.ddp_timeout_minutes <= 0:
        raise ValueError('--ddp_timeout_minutes must be positive')
    args.val_qp_values = _parse_qp_values(args.val_qps, '--val_qps')
    args.ref_qp_values = _parse_qp_values(args.ref_qps, '--ref_qps')


def eager_stance():
    """Force compiled regions back to eager for the duration of the block.

    Validation feeds full-resolution images of varying sizes, which would make
    the enc/dec compiled under ``--compile parts`` recompile for every distinct
    shape. Training keeps its own compiled graphs -- the stance is temporary.
    """
    try:
        return torch.compiler.set_stance('force_eager')
    except Exception as e:  # noqa: BLE001 - older torch without set_stance
        LOGGER.warning('torch.compiler.set_stance unavailable (%s)', e)
        return contextlib.nullcontext()


def psnr_from_mse(mse):
    """PSNR in dB for a mean-squared error measured on data in [0, 1].

    MSE is shift-invariant, so it may be computed on the -0.5-shifted YCbCr
    tensors directly.
    """
    mse = float(mse)
    if not math.isfinite(mse):
        return float('nan')
    if mse <= 0:
        return 99.0
    return -10.0 * math.log10(mse)


@torch.inference_mode()
def validate(net, val_loader, val_qps, lambdas, device, max_side, desc='val'):
    """Evaluate held-out full images at each QP: estimated bpp + real PSNR.

    Adapts the reference test_epoch to the official DMCI API
    (forward_one_frame(x, qp) -> {x_hat, bits_y, bits_z, ...}). bpp uses the
    entropy-model bit estimate (the real range coder needs the CUDA ext, which
    is inference-only); bits are normalized by the ORIGINAL pixel count. Images
    are padded to a multiple of 64, run, then cropped back before PSNR. Returns
    a list of per-QP RD/stat dicts sorted by the input QP order.
    """
    net.eval()
    per_qp = []
    total = len(val_qps) * len(val_loader)
    with tqdm(total=total, desc=desc, dynamic_ncols=True, mininterval=1.0,
              leave=False) as pbar:
        for qp in val_qps:
            qp_t = torch.tensor([int(qp)], dtype=torch.int32, device=device)
            acc = {
                'bpp': 0.0,
                'psnr': 0.0,
                'psnr_y': 0.0,
                'psnr_u': 0.0,
                'psnr_v': 0.0,
            }
            n = 0
            skipped_oom = 0
            skipped_nonfinite = 0
            for x in val_loader:  # batch_size == 1 (images vary in size)
                try:
                    x = x.to(device, non_blocking=True)
                    _, _, H, W = x.shape

                    if max_side and (H > max_side or W > max_side):
                        capped_h = min(H, max_side)
                        capped_w = min(W, max_side)
                        th = (capped_h // 64) * 64 if capped_h >= 64 else H
                        tw = (capped_w // 64) * 64 if capped_w >= 64 else W
                        top, left = (H - th) // 2, (W - tw) // 2
                        x = x[:, :, top:top + th, left:left + tw].contiguous()
                        H, W = th, tw

                    pad_h = (64 - H % 64) % 64
                    pad_w = (64 - W % 64) % 64
                    x_pad = (
                        F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
                        if (pad_h or pad_w) else x
                    )

                    try:
                        out = net.forward_one_frame(x_pad, qp_t)
                    except torch.cuda.OutOfMemoryError:
                        skipped_oom += 1
                        torch.cuda.empty_cache()
                        continue

                    x_hat = out['x_hat'][:, :, :H, :W].clamp(-0.5, 0.5)
                    bits = (out['bits_y'] + out['bits_z']).sum().float()
                    channel_mse = (x_hat - x).pow(2).mean(dim=(0, 2, 3))
                    if not bool(
                        torch.isfinite(bits).item()
                        and torch.isfinite(channel_mse).all().item()
                    ):
                        skipped_nonfinite += 1
                        continue

                    acc['bpp'] += float((bits / (H * W)).item())
                    acc['psnr'] += psnr_from_mse(channel_mse.mean().item())
                    acc['psnr_y'] += psnr_from_mse(channel_mse[0].item())
                    acc['psnr_u'] += psnr_from_mse(channel_mse[1].item())
                    acc['psnr_v'] += psnr_from_mse(channel_mse[2].item())
                    n += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix(
                        qp=int(qp),
                        oom=skipped_oom,
                        nonfinite=skipped_nonfinite,
                        refresh=False,
                    )

            if n == 0:
                raise RuntimeError(
                    f'validation produced no valid images at QP {int(qp)} '
                    f'(oom={skipped_oom}, nonfinite={skipped_nonfinite})'
                )
            per_qp.append({
                'qp': int(qp),
                'lmbda': float(lambdas[int(qp)]),
                'bpp': acc['bpp'] / n,
                'psnr': acc['psnr'] / n,
                'psnr_y': acc['psnr_y'] / n,
                'psnr_u': acc['psnr_u'] / n,
                'psnr_v': acc['psnr_v'] / n,
                'num_images': n,
                'skipped_oom': skipped_oom,
                'skipped_nonfinite': skipped_nonfinite,
            })
    return per_qp


def _load_json(path):
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning('failed to load %s: %s', path, e)
            return None
    return None


def _atomic_write(path, write_to):
    """Write through a temp file in the same directory, then rename into place.

    ``write_to`` is handed the temp path. The rename is atomic, so a crash
    mid-write can never leave a truncated file where a reader expects a complete
    one. Returns False and warns rather than raising: losing a sidecar must not
    kill a 105-epoch run.
    """
    tmp_path = None
    try:
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(
            dir=directory,
            prefix=f'.{os.path.basename(path)}.',
            suffix='.tmp',
        )
        os.close(fd)
        write_to(tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to save %s: %s', path, e)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return False
    return True


def _save_json(path, obj):
    def write_to(tmp_path):
        with open(tmp_path, 'w') as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_write(path, write_to)


def _atomic_torch_save(obj, path):
    return _atomic_write(path, lambda tmp_path: torch.save(obj, tmp_path))


def _load_best_checkpoint_metadata(path, expected_identity):
    """Read and validate the small metadata fields from a best checkpoint."""
    if not path or not os.path.isfile(path):
        return None
    try:
        checkpoint = torch.load(
            path,
            map_location=torch.device('cpu'),
            weights_only=True,
            mmap=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError('checkpoint payload is not a dictionary')
        state_dict = checkpoint.get('state_dict')
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError('checkpoint has no usable state_dict')
        if checkpoint.get('validation_identity') != expected_identity:
            raise ValueError('checkpoint validation identity does not match')
        epoch = int(checkpoint['epoch'])
        score = float(checkpoint['val_mean_psnr'])
        if epoch < 0 or not math.isfinite(score):
            raise ValueError('checkpoint epoch/score is invalid')
        return {'epoch': epoch, 'score': score}
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to validate best checkpoint %s: %s', path, e)
        return None


def _file_identity(path):
    if not path:
        return None
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
    except OSError:
        return {'path': absolute, 'missing': True}
    return {
        'path': absolute,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _sanitize_rd_points(points):
    """Keep only complete, finite RD points with at least one valid image."""
    required = ('bpp', 'psnr', 'psnr_y', 'psnr_u', 'psnr_v')
    sanitized = {}
    if not isinstance(points, list):
        return []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            qp = int(point['qp'])
            values = {key: float(point[key]) for key in required}
            num_images = int(point.get('num_images', 1))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= qp < DMCI.qp_num() or num_images <= 0:
            continue
        if not all(math.isfinite(value) for value in values.values()):
            continue
        clean = dict(point)
        clean.update(values)
        clean['qp'] = qp
        clean['num_images'] = num_images
        try:
            clean['skipped_oom'] = int(point.get('skipped_oom', 0))
            clean['skipped_nonfinite'] = int(point.get('skipped_nonfinite', 0))
        except (TypeError, ValueError):
            clean['skipped_oom'] = 0
            clean['skipped_nonfinite'] = 0
        sanitized[qp] = clean
    return [sanitized[qp] for qp in sorted(sanitized)]


def _sanitize_rd_history(history):
    by_epoch = {}
    if not isinstance(history, list):
        return []
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            epoch = int(item['epoch'])
        except (KeyError, TypeError, ValueError):
            continue
        points = _sanitize_rd_points(item.get('points'))
        if points:
            by_epoch[epoch] = {'epoch': epoch, 'points': points}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _best_history_psnr(history):
    best_epoch = None
    best_score = float('-inf')
    for item in history:
        scores = [point['psnr'] for point in item['points']]
        if not scores:
            continue
        score = sum(scores) / len(scores)
        if score > best_score:
            best_epoch = int(item['epoch'])
            best_score = score
    return best_epoch, best_score


def _upsert_rd_history(history, epoch, points):
    """Replace an existing epoch entry and clean duplicate epochs from old history."""
    by_epoch = {}
    for item in _sanitize_rd_history(history):
        by_epoch[item['epoch']] = item
    clean_points = _sanitize_rd_points(points)
    if clean_points:
        by_epoch[int(epoch)] = {'epoch': int(epoch), 'points': clean_points}
    return [by_epoch[key] for key in sorted(by_epoch)]


def save_channel_rd_plots(history, current_points, reference, out_dir, epoch):
    """Save current-epoch and all-epochs RD figures for Y / Cb / Cr.

    Current-epoch figures use blue for the trained model and red for the
    official reference. All-epochs figures color validation epochs with
    viridis and retain the red official reference. The latter use a colorbar,
    while neither view needs a legend.

    Returns ``{'current': {channel_key: path}, 'all_epochs': {...}}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    mpl_config_dir = os.path.join(out_dir, '.matplotlib')
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', mpl_config_dir)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('matplotlib unavailable (%s); skipping RD plots', e)
        return {'current': {}, 'all_epochs': {}}
    current_points = _sanitize_rd_points(current_points)
    reference = _sanitize_rd_points(reference)
    history = _sanitize_rd_history(history)
    if not current_points or not history:
        return {'current': {}, 'all_epochs': {}}

    epochs = [int(h['epoch']) for h in history]
    emin, emax = min(epochs), max(epochs)
    cnorm = mcolors.Normalize(vmin=emin, vmax=emax if emax > emin else emin + 1)
    channels = [('psnr_y', 'Y'), ('psnr_u', 'Cb'), ('psnr_v', 'Cr')]

    paths = {'current': {}, 'all_epochs': {}}
    for key, label in channels:
        ref = sorted(reference or [], key=lambda p: p['bpp'])
        current = sorted(current_points, key=lambda p: p['bpp'])

        # One durable image for this validation epoch.
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        if ref:
            ax.plot(
                [p['bpp'] for p in ref],
                [p[key] for p in ref],
                marker='*',
                ms=8,
                lw=1.5,
                color='tab:red',
                zorder=3,
            )
        ax.plot(
            [p['bpp'] for p in current],
            [p[key] for p in current],
            marker='o',
            ms=4,
            lw=1.3,
            color='tab:blue',
            zorder=4,
        )
        ax.set_xlabel('bpp')
        ax.set_ylabel(f'PSNR-{label} (dB)')
        ax.set_title(f'Val RD: PSNR-{label} vs bpp — epoch {epoch}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        current_path = os.path.join(out_dir, f'val_rd_epoch_{int(epoch):03d}_{key}.png')
        fig.savefig(current_path, dpi=110)
        plt.close(fig)
        paths['current'][key] = current_path

        # One rolling image containing the complete validation trajectory.
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        if ref:
            ax.plot(
                [p['bpp'] for p in ref],
                [p[key] for p in ref],
                marker='*',
                ms=8,
                lw=1.5,
                color='tab:red',
                zorder=5,
            )
        for h in history:
            pts = sorted(h['points'], key=lambda p: p['bpp'])
            ax.plot(
                [p['bpp'] for p in pts],
                [p[key] for p in pts],
                marker='o',
                ms=4,
                lw=1.0,
                color=cm.viridis(cnorm(int(h['epoch']))),
            )
        ax.set_xlabel('bpp')
        ax.set_ylabel(f'PSNR-{label} (dB)')
        ax.set_title(f'Val RD: PSNR-{label} vs bpp — all epochs')
        ax.grid(True, alpha=0.3)
        sm = cm.ScalarMappable(norm=cnorm, cmap='viridis')
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='train epoch')
        fig.tight_layout()
        history_path = os.path.join(out_dir, f'val_rd_all_epochs_{key}.png')
        fig.savefig(history_path, dpi=110)
        plt.close(fig)
        paths['all_epochs'][key] = history_path
    return paths


def compute_reference_rd(ref_ckpt, val_loader, ref_qps, lambdas, device, max_side, cache_path,
                         cache_identity):
    """Evaluate the official model once to get its reference RD points (cached).

    The cache is keyed by ``cache_identity`` (checkpoint file, validation
    configuration, QPs and lambdas), so a single ``cache_path`` may be shared by
    every run that validates the same way -- the official model then runs once
    in total rather than once per ``save_dir``.

    Returns ``(points, status)`` where status describes whether the reference
    was disabled, missing, loaded from a matching cache, or freshly computed.
    """
    if not ref_ckpt:
        return None, 'disabled'
    if not os.path.exists(ref_ckpt):
        LOGGER.warning('--ref_ckpt %s not found; plots will omit the reference curve', ref_ckpt)
        return None, 'missing'

    cached = _load_json(cache_path)
    if isinstance(cached, dict) and cached.get('identity') == cache_identity:
        cached_points = _sanitize_rd_points(cached.get('points'))
        if [p['qp'] for p in cached_points] == ref_qps:
            return cached_points, 'cached'

    # Always the official architecture: the reference weights are the official
    # pretrained ones, independent of the architecture being trained.
    ref_net = DMCI()
    ref_net.load_state_dict(get_state_dict(ref_ckpt))
    ref_net = ref_net.to(device).eval()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    reference = validate(
        ref_net,
        val_loader,
        ref_qps,
        lambdas,
        device,
        max_side,
        desc='official reference',
    )
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    del ref_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        LOGGER.warning('failed to create reference cache directory %s: %s', cache_dir, e)
    cache_saved = _save_json(cache_path, {
        'identity': cache_identity,
        'points': reference,
    })
    return reference, 'computed' if cache_saved else 'computed-cache-write-failed'


def _update_wandb_summary(wandb_run, values):
    """Best-effort summary metadata without making monitoring training-critical."""
    if wandb_run is None or getattr(wandb_run, 'run', None) is None:
        return
    try:
        for key, value in values.items():
            wandb_run.run.summary[key] = value
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to update wandb summary: %s', e)


def _wandb_log(wandb_run, values):
    if wandb_run is None:
        return False
    try:
        wandb_run.log(values)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('wandb.log failed; continuing training without this record: %s', e)
        return False
    return True


def _wandb_image(wandb_run, path):
    if wandb_run is None:
        return None
    try:
        return wandb_run.Image(path)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to create wandb image for %s: %s', path, e)
        return None


def _wandb_finish(wandb_run):
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('wandb.finish failed: %s', e)


def _log_reference_table(wandb_run, reference):
    if wandb_run is None or not reference:
        return
    columns = [
        'qp', 'lambda', 'bpp', 'psnr', 'psnr_y', 'psnr_u', 'psnr_v',
        'num_images', 'skipped_oom', 'skipped_nonfinite',
    ]
    data = [
        [
            p['qp'], p.get('lmbda'), p['bpp'], p['psnr'], p['psnr_y'],
            p['psnr_u'], p['psnr_v'], p['num_images'], p.get('skipped_oom', 0),
            p.get('skipped_nonfinite', 0),
        ]
        for p in reference
    ]
    try:
        # Static reference metadata belongs in the run summary. Logging it as
        # a history row would create a second row at ``begin_epoch``.
        _update_wandb_summary(wandb_run, {
            'reference/rd_points': wandb_run.Table(data=data, columns=columns),
        })
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to log official reference table to wandb: %s', e)


def maybe_init_wandb(args, rank, extra_config):
    """Init wandb on the main process only, degrading gracefully if unavailable.

    Returns the wandb module (used for .log/.finish) or None when monitoring is
    disabled, we are on a non-main DDP rank, or wandb is not installed / fails
    to start. Training never depends on the return value.
    """
    if rank > 0 or args.wandb_mode == 'disabled':
        return None
    try:
        import wandb
    except ImportError:
        LOGGER.warning(
            'wandb not installed; install it to enable monitoring; continuing without it'
        )
        return None

    exp_name = args.exp_name or os.path.basename(os.path.normpath(args.save_dir))
    config = {**vars(args), **extra_config}
    safe_exp_name = ''.join(
        char if char.isalnum() or char in '._-' else '_' for char in exp_name
    )
    run_id_path = os.path.join(args.save_dir, f'wandb_run_id_{safe_exp_name}.txt')
    init_kwargs = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'name': exp_name,
        'mode': args.wandb_mode,
        'config': config,
        'dir': args.save_dir,
    }
    if extra_config.get('begin_epoch', 0) > 0 and os.path.exists(run_id_path):
        try:
            with open(run_id_path) as f:
                previous_id = f.read().strip()
            if previous_id:
                init_kwargs.update(id=previous_id, resume='allow')
        except OSError as e:
            LOGGER.warning('failed to read wandb run id from %s: %s', run_id_path, e)
    try:
        wandb.init(**init_kwargs)
    except Exception as e:  # noqa: BLE001 - never let monitoring break training
        LOGGER.warning('wandb.init failed (%s); continuing without monitoring', e)
        return None
    try:
        with open(run_id_path, 'w') as f:
            f.write(str(wandb.run.id))
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('failed to persist wandb run id to %s: %s', run_id_path, e)

    # epoch-indexed curves for per-epoch metrics; optimizer-step-indexed curves
    # for the within-epoch train-step/* snapshots. define_metric only accepts
    # suffix globs, so use a "*" catch-all keyed to epoch and override just the
    # train-step/* namespace. Wrapped so a wandb-version quirk can't kill it.
    try:
        wandb.define_metric('epoch')
        wandb.define_metric('total_optimizer_steps')
        # Register the more-specific glob first. Some wandb versions select
        # the first matching pattern, so a leading "*" would steal train-step.
        wandb.define_metric('train-step/*', step_metric='total_optimizer_steps')
        wandb.define_metric('*', step_metric='epoch')
    except Exception as e:  # noqa: BLE001
        LOGGER.warning('wandb.define_metric skipped (%s); metrics are still logged', e)
        _update_wandb_summary(wandb, {'monitoring/define_metric_error': str(e)})
    return wandb


def train_one_epoch(i_net, dataloader, optimizer, epoch, rank, wandb_run, global_step,
                    step_log_interval, use_amp=False):
    training_strategy = get_training_strategy()
    i_net.train()
    device = next(i_net.parameters()).device
    # bf16 keeps fp32's exponent range, so the entropy model's small
    # probabilities survive the half-precision round trip and no GradScaler is
    # needed -- gradient clipping and the non-finite skip below work unchanged.
    amp_enabled = use_amp and device.type == 'cuda'

    idx = min(len(training_strategy) - 1, epoch)
    _, lr, patch_width, patch_height = training_strategy[idx]

    for g in optimizer.param_groups:
        g['lr'] = lr

    world_size = dist.get_world_size() if rank >= 0 else 1
    # The patch size is owned by train(): with persistent workers a mutation
    # here would never reach the worker processes, so the loader is rebuilt
    # there whenever the schedule changes it.

    # loss/grad_norm cover every successful batch. Rate/distortion metrics are
    # sampled every 200 dataloader batches and at each train-step log point,
    # avoiding full loss-info extraction on every batch. DDP sums are reduced
    # across ranks before W&B logging.
    ep = {
        'loss_sum': 0.0,
        'loss_n': 0,
        'sample_n': 0,
        'gn_sum': 0.0,
        'gn_max': 0.0,
        'bpp_sum': 0.0,
        'bpp_y_sum': 0.0,
        'bpp_z_sum': 0.0,
        'mse_sum': 0.0,
        'mse_yuv_sum': 0.0,
        'mse_y_sum': 0.0,
        'mse_u_sum': 0.0,
        'mse_v_sum': 0.0,
        'info_n': 0,
        'skipped': 0,
    }
    track_mem = device.type == 'cuda'
    if track_mem:
        torch.cuda.reset_peak_memory_stats(device)
    epoch_t0 = time.time()

    progress = tqdm(
        dataloader,
        desc=f'train epoch {epoch}',
        dynamic_ncols=True,
        mininterval=1.0,
        disable=rank > 0,
    )
    latest_info = None
    for i, batch in enumerate(progress):
        # The loader pins its batches, so the copy can overlap with the tail of
        # the previous step instead of blocking the host.
        batch = [t.to(device, non_blocking=True) for t in batch]
        batch_size = batch[0].size(0)
        qp = batch[-2]
        curr_lambdas = batch[-1]

        # Check the successful step number that this batch would produce. This
        # logs 100, 200, ... rather than the previous off-by-one 1, 101, ...
        do_step_snapshot = (
            step_log_interval > 0
            and (global_step + 1) % step_log_interval == 0
        )
        do_wandb_step = wandb_run is not None and do_step_snapshot
        do_monitor_info = i % TRAIN_METRIC_MONITOR_INTERVAL == 0
        need_info = do_monitor_info or do_step_snapshot

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            loss, info = i_net(batch[0], qp, lambdas=curr_lambdas, get_loss_info=need_info)
        optimizer.zero_grad()
        loss.backward()

        # One device->host sync per step: the clipped norm (needed on the host
        # to decide whether to skip the step) is read together with the loss.
        grad_norm_t = clip_grad_norm_(i_net.parameters(), max_norm=0.1,
                                      error_if_nonfinite=False)
        total_norm, loss_value = torch.stack(
            (grad_norm_t.detach().float(), loss.detach().float())
        ).tolist()
        if math.isnan(total_norm) or math.isinf(total_norm):
            ep['skipped'] += 1
            optimizer.zero_grad(set_to_none=True)
            if rank <= 0:
                progress.set_postfix(
                    loss=f"{ep['loss_sum'] / max(ep['loss_n'], 1):.4f}",
                    lr=f"{lr:.1e}",
                    skipped=ep['skipped'],
                    refresh=False,
                )
            continue

        optimizer.step()
        global_step += 1

        ep['loss_sum'] += loss_value
        ep['loss_n'] += 1
        ep['sample_n'] += batch_size
        ep['gn_sum'] += total_norm
        ep['gn_max'] = max(ep['gn_max'], total_norm)
        if info is not None:
            s = info['scalars']
            latest_info = s
            ep['bpp_sum'] += s['bpp']
            ep['bpp_y_sum'] += s['bpp_y']
            ep['bpp_z_sum'] += s['bpp_z']
            ep['mse_sum'] += s['mse']
            ep['mse_yuv_sum'] += s['mse_yuv']
            ep['mse_y_sum'] += s['mse_y']
            ep['mse_u_sum'] += s['mse_u']
            ep['mse_v_sum'] += s['mse_v']
            ep['info_n'] += 1

        if do_step_snapshot:
            s = info['scalars']
            step_values = torch.tensor(
                [
                    s['loss'], s['bpp'], s['bpp_y'], s['bpp_z'], s['mse'],
                    s['mse_yuv'], s['mse_y'], s['mse_u'], s['mse_v'], total_norm,
                ],
                dtype=torch.float64,
                device=device,
            )
            if rank >= 0:
                dist.all_reduce(step_values, op=dist.ReduceOp.SUM)
                step_values /= world_size
            if do_wandb_step:
                values = step_values.tolist()
                _wandb_log(wandb_run, {
                    'train-step/loss': values[0],
                    'train-step/bpp': values[1],
                    'train-step/bpp_y': values[2],
                    'train-step/bpp_z': values[3],
                    'train-step/mse': values[4],
                    'train-step/mse_yuv': values[5],
                    'train-step/psnr': psnr_from_mse(values[5]),
                    'train-step/psnr_y': psnr_from_mse(values[6]),
                    'train-step/psnr_u': psnr_from_mse(values[7]),
                    'train-step/psnr_v': psnr_from_mse(values[8]),
                    'train-step/grad_norm': values[9],
                    'train-step/lr': optimizer.param_groups[0]['lr'],
                    'total_optimizer_steps': global_step,
                })

        if rank <= 0:
            if i % 10 == 0 or i + 1 == len(dataloader):
                postfix = {
                    'loss': f"{ep['loss_sum'] / max(ep['loss_n'], 1):.4f}",
                    'lr': f'{lr:.1e}',
                    'skipped': ep['skipped'],
                }
                if latest_info is not None:
                    postfix['bpp'] = f"{latest_info['bpp']:.4f}"
                    postfix['psnr_y'] = f"{psnr_from_mse(latest_info['mse_y']):.2f}"
                progress.set_postfix(postfix, refresh=False)

    sum_keys = (
        'loss_sum', 'loss_n', 'sample_n', 'gn_sum', 'bpp_sum', 'bpp_y_sum',
        'bpp_z_sum', 'mse_sum', 'mse_yuv_sum', 'mse_y_sum', 'mse_u_sum',
        'mse_v_sum', 'info_n',
    )
    reduced = torch.tensor(
        [ep[key] for key in sum_keys],
        dtype=torch.float64,
        device=device,
    )
    extrema = torch.tensor(
        [ep['gn_max'], ep['skipped']],
        dtype=torch.float64,
        device=device,
    )
    if rank >= 0:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        dist.all_reduce(extrema, op=dist.ReduceOp.MAX)
    global_ep = dict(zip(sum_keys, reduced.tolist()))
    global_ep['gn_max'], global_ep['skipped'] = extrema.tolist()

    stats = None
    if rank <= 0:
        loss_n = max(global_ep['loss_n'], 1)
        info_n = max(global_ep['info_n'], 1)
        epoch_time = max(time.time() - epoch_t0, 1e-9)
        stats = {
            'loss': global_ep['loss_sum'] / loss_n,
            'grad_norm': global_ep['gn_sum'] / loss_n,
            'grad_norm_max': global_ep['gn_max'],
            'bpp': global_ep['bpp_sum'] / info_n,
            'bpp_y': global_ep['bpp_y_sum'] / info_n,
            'bpp_z': global_ep['bpp_z_sum'] / info_n,
            'mse': global_ep['mse_sum'] / info_n,
            'mse_yuv': global_ep['mse_yuv_sum'] / info_n,
            'mse_y': global_ep['mse_y_sum'] / info_n,
            'mse_u': global_ep['mse_u_sum'] / info_n,
            'mse_v': global_ep['mse_v_sum'] / info_n,
            'metric_batches': int(global_ep['info_n'] / world_size),
            'lr': lr,
            'patch_size': patch_width,
            'skipped': int(global_ep['skipped']),
            'epoch_time': epoch_time,
            'samples_per_sec': global_ep['sample_n'] / epoch_time,
            'max_gpu_mem_mb': (torch.cuda.max_memory_allocated(device) / 1e6
                               if track_mem else None),
        }
    return global_step, stats


def train(rank, args):
    world_size = init_train(rank, args.save_dir, args.ddp_timeout_minutes)
    # Seed before the model exists so its initial weights are reproducible too.
    # Every rank shares the seed: DDP broadcasts rank 0's weights anyway, and
    # the per-rank data split comes from DistributedSampler, not the RNG.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        # init_train() turned the cuDNN autotuner on; it picks algorithms from
        # measured timings, which alone makes two runs of the same seed diverge.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if args.seed is None:
            LOGGER.warning('--deterministic without --seed still randomizes init and data')
    lambdas = get_training_lambdas(args.lambdas, DMCI.qp_num())
    train_dataset = ImageFolder(args.train_dataset, 256, 256, DMCI.qp_num(), lambdas,
                                fixed_qp=args.train_qp)

    i_net = DMCI()
    # Loss-only knobs; they touch no parameter, so a run can switch between the
    # official loss and any of these variants on resume.
    i_net.set_mse_config(rgb_weight=args.mse_rgb_weight,
                         yuv_mean=args.mse_yuv_mean,
                         yuv_weights=args.mse_yuv_weight_values)
    if rank <= 0:
        w_y, w_u, w_v = args.mse_yuv_weight_values
        LOGGER.info('training distortion: %.3g*YUV_%s(%g:%g:%g) + %.3g*RGB',
                    1.0 - args.mse_rgb_weight, args.mse_yuv_mean,
                    w_y, w_u, w_v, args.mse_rgb_weight)
    begin_epoch, opt_status = load_existing_weights(
        args.save_dir, i_net, rank, verbose=False
    )
    monitor_state_path = os.path.join(args.save_dir, 'training_monitor_state.json')
    monitor_state = _load_json(monitor_state_path)
    restored_global_step = None
    monitor_best_valid = False
    best_psnr = float('-inf')
    best_epoch = None
    if isinstance(monitor_state, dict):
        try:
            state_epoch = int(monitor_state['epoch'])
            if state_epoch == begin_epoch - 1:
                candidate_global_step = int(monitor_state['global_step'])
                if candidate_global_step < 0:
                    raise ValueError('negative global step')
                restored_global_step = candidate_global_step
                monitor_best_valid = True
                state_best = monitor_state.get('best_psnr')
                if state_best is not None and math.isfinite(float(state_best)):
                    best_psnr = float(state_best)
                    state_best_epoch = monitor_state.get('best_epoch')
                    best_epoch = int(state_best_epoch) if state_best_epoch is not None else None
        except (KeyError, TypeError, ValueError):
            restored_global_step = None
            monitor_best_valid = False

    device = get_current_device(rank, verbose=False)
    # get_current_device returns a string such as 'cuda:0', not a torch.device.
    if args.amp and str(device).startswith('cuda'):
        # torch.autocast(dtype=bfloat16) raises outright when torch reports no
        # bf16 for the device, so --amp would kill the run on its first batch.
        # Degrade instead: an unavailable accelerant must not cost 105 epochs.
        if not torch.cuda.is_bf16_supported():
            LOGGER.warning('this GPU has no bf16 support at all; disabling --amp '
                           'and training in fp32')
            args.amp = False
        # is_bf16_supported() also answers True when bf16 merely *works* --
        # before sm_80 there are no bf16 tensor cores, so the math is emulated.
        # That trains correctly but can be slower than fp32, and the permissive
        # check above would never say so.
        elif torch.cuda.get_device_properties(device).major < 8:
            LOGGER.warning('bf16 is emulated on this GPU (compute capability < 8.0); '
                           '--amp may be slower than fp32 here -- benchmark before '
                           'keeping it')
    if rank >= 0:
        i_net = i_net.cuda(rank)
    else:
        i_net = i_net.to(device)
    i_net = wrap_ddp(i_net, rank)

    # Only the training step goes through the compiled wrapper. Validation,
    # save_status/save_ckpt and the best-checkpoint writer keep using ``i_net``,
    # whose state_dict has no '_orig_mod.' prefix, so checkpoints stay
    # interchangeable between compiled and eager runs.
    train_net = i_net
    if args.compile == 'full':
        train_net = torch.compile(i_net, mode=args.compile_mode)
    elif args.compile == 'parts':
        # Compiling the bound forward (rather than replacing the submodule with
        # an OptimizedModule) keeps the module tree - and therefore the
        # state_dict keys - untouched.
        base_net = i_net.module if isinstance(i_net, DDP) else i_net
        base_net.enc.forward = torch.compile(base_net.enc.forward, mode=args.compile_mode)
        base_net.dec.forward = torch.compile(base_net.dec.forward, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(
        i_net.parameters(), lr=1e-4,
        fused=args.fused_adam and str(device).startswith('cuda'))
    if opt_status is not None:
        optimizer.load_state_dict(opt_status)

    data_generator = torch.Generator()
    if args.seed is not None:
        data_generator.manual_seed(args.seed)

    def build_dataloader(patch_width, patch_height):
        """Recreate the loader so persistent workers pick up the new patch size."""
        train_dataset.set_patch_size(patch_width, patch_height)
        return get_dataloader(train_dataset, rank, world_size, args.batch_size,
                              args.num_workers, generator=data_generator,
                              seed=args.seed if args.seed is not None else 0)

    strategy = get_training_strategy()

    def patch_size_for(epoch):
        _, _, patch_width, patch_height = strategy[min(len(strategy) - 1, epoch)]
        return patch_width, patch_height

    current_patch = patch_size_for(begin_epoch)
    dataloader = build_dataloader(*current_patch)

    steps_per_epoch = len(dataloader)
    fallback_global_step = begin_epoch * steps_per_epoch
    global_step = (
        restored_global_step
        if restored_global_step is not None and restored_global_step >= 0
        else fallback_global_step
    )
    wandb_run = maybe_init_wandb(args, rank, extra_config={
        'model_class': f'{DMCI.__module__}.{DMCI.__qualname__}',
        'qp_num': DMCI.qp_num(),
        'lambda_min': float(lambdas[0]),
        'lambda_max': float(lambdas[-1]),
        'mse_rgb_weight': float(args.mse_rgb_weight),
        'mse_yuv_weight': float(1.0 - args.mse_rgb_weight),
        'mse_yuv_mean': args.mse_yuv_mean,
        'mse_yuv_weights': list(args.mse_yuv_weight_values),
        'model_params': int(sum(p.numel() for p in i_net.parameters())),
        'world_size': world_size,
        'steps_per_epoch': steps_per_epoch,
        'train_metric_monitor_interval': TRAIN_METRIC_MONITOR_INTERVAL,
        'train_size': len(train_dataset),
        'begin_epoch': begin_epoch,
        'device': str(device),
        'cuda_device_count': torch.cuda.device_count(),
    })
    _update_wandb_summary(wandb_run, {
        'checkpoint/resumed': bool(begin_epoch > 0),
        'checkpoint/begin_epoch': begin_epoch,
    })

    # ---- validation setup (main process only) ----
    val_loader = None
    val_qps = args.val_qp_values
    reference_rd = None
    rd_history = []
    rd_history_identity = (
        monitor_state.get('validation_identity')
        if monitor_best_valid and isinstance(monitor_state, dict)
        else None
    )
    rd_history_path = os.path.join(args.save_dir, 'val_rd_history.json')
    rd_plot_dir = os.path.join(args.save_dir, 'rd_plots')
    best_path = (
        monitor_state.get('best_checkpoint_path')
        if monitor_best_valid and isinstance(monitor_state, dict)
        else None
    )
    if rank <= 0 and args.val_dataset:
        val_set = ImageValFolder(args.val_dataset, args.val_list)
        if len(val_set) == 0:
            raise ValueError('validation dataset is empty')
        val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers,
                                shuffle=False, pin_memory=True)
        val_list_path = args.val_list or os.path.join(args.val_dataset, 'description.json')
        validation_identity = {
            'dataset_root': os.path.abspath(args.val_dataset),
            'list': _file_identity(val_list_path),
            'max_side': args.val_max_side,
        }
        rd_history_identity = {
            'version': 2,
            'validation': validation_identity,
            'qps': val_qps,
        }
        validation_signature = hashlib.sha256(
            json.dumps(rd_history_identity, sort_keys=True).encode()
        ).hexdigest()[:10]
        best_path = os.path.join(
            args.save_dir, f'ckpt_best_{validation_signature}.pth.tar'
        )
        if (
            monitor_best_valid
            and monitor_state.get('validation_identity') != rd_history_identity
        ):
            monitor_best_valid = False
            best_epoch = None
            best_psnr = float('-inf')
            _update_wandb_summary(wandb_run, {
                'best/epoch': None,
                'best/checkpoint_path': '',
            })

        # Keep histories from different validation configurations separate.
        loaded_hist = _load_json(rd_history_path)
        loaded_entries = []
        if isinstance(loaded_hist, dict):
            if loaded_hist.get('identity') == rd_history_identity:
                loaded_entries = loaded_hist.get('entries')
            else:
                rd_history_path = os.path.join(
                    args.save_dir, f'val_rd_history_{validation_signature}.json'
                )
                rd_plot_dir = os.path.join(
                    args.save_dir, f'rd_plots_{validation_signature}'
                )
                loaded_hist = _load_json(rd_history_path)
                loaded_entries = (
                    loaded_hist.get('entries')
                    if isinstance(loaded_hist, dict)
                    and loaded_hist.get('identity') == rd_history_identity
                    else []
                )

        rd_history = [
            item for item in _sanitize_rd_history(loaded_entries)
            if item['epoch'] < begin_epoch
            and [point['qp'] for point in item['points']] == val_qps
        ]
        history_best_epoch, history_best_psnr = _best_history_psnr(rd_history)
        if (
            not monitor_best_valid
            or best_epoch is None
            or not math.isfinite(best_psnr)
        ):
            if history_best_psnr > best_psnr:
                best_epoch = history_best_epoch
                best_psnr = history_best_psnr

        # The namespaced checkpoint is authoritative when present, and also
        # lets recovery survive a missing monitor/history sidecar.
        best_metadata = _load_best_checkpoint_metadata(best_path, rd_history_identity)
        if best_metadata is not None:
            best_epoch = best_metadata['epoch']
            best_psnr = best_metadata['score']
        elif math.isfinite(best_psnr):
            best_epoch = None
            best_psnr = float('-inf')

        _update_wandb_summary(wandb_run, {
            'validation/num_images': len(val_set),
            'validation/qps': val_qps,
            'validation/interval': args.val_interval,
            'validation/max_side': args.val_max_side,
            'validation/rd_history_path': rd_history_path,
            'validation/rd_plot_dir': rd_plot_dir,
            'validation/signature': validation_signature,
        })
        if best_epoch is not None:
            _update_wandb_summary(wandb_run, {
                'best/epoch': best_epoch,
                'best/checkpoint_path': best_path,
            })
        ref_qps = args.ref_qp_values
        ref_cache_path = args.ref_cache or os.path.join(args.save_dir, 'ref_rd_points.json')
        reference_rd, reference_status = compute_reference_rd(
            args.ref_ckpt,
            val_loader,
            ref_qps,
            lambdas,
            device,
            args.val_max_side,
            ref_cache_path,
            cache_identity={
                'version': 2,
                'checkpoint': _file_identity(args.ref_ckpt),
                'validation': validation_identity,
                'qps': ref_qps,
                'lambdas': [float(lambdas[qp]) for qp in ref_qps],
            },
        )
        _update_wandb_summary(wandb_run, {
            'reference/status': reference_status,
            'reference/checkpoint': args.ref_ckpt or '',
            'reference/cache_path': ref_cache_path,
            'reference/num_points': len(reference_rd or []),
        })
        _log_reference_table(wandb_run, reference_rd)

    if rank >= 0:
        dist.barrier()

    for epoch in range(begin_epoch, args.epochs):
        # Reseed per epoch so a given (seed, epoch) always yields the same data
        # order, independent of how many epochs this process has already run.
        if args.seed is not None:
            data_generator.manual_seed(args.seed + 10007 * epoch)
        epoch_patch = patch_size_for(epoch)
        if epoch_patch != current_patch:
            current_patch = epoch_patch
            dataloader = build_dataloader(*current_patch)
        if rank >= 0:
            dataloader.sampler.set_epoch(epoch)
        global_step, stats = train_one_epoch(train_net, dataloader, optimizer, epoch, rank,
                                              wandb_run, global_step, args.wandb_step_interval,
                                              use_amp=args.amp)

        if rank <= 0:
            epoch_log = None
            validation_log = None
            if wandb_run is not None and stats is not None:
                epoch_log = {
                    'epoch': epoch,
                    'train-loss/total': stats['loss'],
                    'train-rate/bpp': stats['bpp'],
                    'train-rate/bpp_y': stats['bpp_y'],
                    'train-rate/bpp_z': stats['bpp_z'],
                    'train-distortion/mse': stats['mse'],
                    'train-distortion/mse_yuv': stats['mse_yuv'],
                    'train-distortion/mse_y': stats['mse_y'],
                    'train-distortion/mse_u': stats['mse_u'],
                    'train-distortion/mse_v': stats['mse_v'],
                    'train-quality/psnr': psnr_from_mse(stats['mse_yuv']),
                    'train-quality/psnr_y': psnr_from_mse(stats['mse_y']),
                    'train-quality/psnr_u': psnr_from_mse(stats['mse_u']),
                    'train-quality/psnr_v': psnr_from_mse(stats['mse_v']),
                    'train-schedule/lr': stats['lr'],
                    'train-schedule/patch_size': stats['patch_size'],
                    'train-optimization/grad_norm': stats['grad_norm'],
                    'train-optimization/grad_norm_max': stats['grad_norm_max'],
                    'train-progress/sampled_metric_batches': stats['metric_batches'],
                    'train-stability/skipped_batches': stats['skipped'],
                    'train-runtime/epoch_time_sec': stats['epoch_time'],
                    'train-runtime/samples_per_sec': stats['samples_per_sec'],
                }
                if stats['max_gpu_mem_mb'] is not None:
                    epoch_log['train-runtime/max_gpu_mem_mb'] = stats['max_gpu_mem_mb']

            # ---- validation ----
            do_val = val_loader is not None and (
                (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1)
            net_eval = i_net.module if isinstance(i_net, DDP) else i_net
            if do_val:
                # Deterministic bit-estimation noise so the RD curve is
                # comparable across epochs; restore the training RNG after.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                with eager_stance():
                    per_qp = validate(net_eval, val_loader, val_qps, lambdas, device,
                                      args.val_max_side, desc=f'val epoch {epoch}')
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                i_net.train()

                valid_psnr = [
                    p['psnr'] for p in per_qp
                    if p['num_images'] > 0 and math.isfinite(p['psnr'])
                ]
                mean_psnr = (
                    sum(valid_psnr) / len(valid_psnr)
                    if valid_psnr else float('-inf')
                )

                # Upsert avoids duplicate epochs after a resume/re-run and also
                # cleans duplicates already present in the history JSON.
                rd_history = _upsert_rd_history(rd_history, epoch, per_qp)
                history_saved = _save_json(rd_history_path, {
                    'identity': rd_history_identity,
                    'entries': rd_history,
                })
                try:
                    plot_paths = save_channel_rd_plots(
                        rd_history,
                        per_qp,
                        reference_rd,
                        rd_plot_dir,
                        epoch,
                    )
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning('failed to build validation RD plots: %s', e)
                    plot_paths = {'current': {}, 'all_epochs': {}}
                _update_wandb_summary(wandb_run, {
                    'validation/rd_history_entries': len(rd_history),
                    'validation/rd_history_saved': history_saved,
                    'validation/rd_plots_available': any(
                        plot_paths[view] for view in ('current', 'all_epochs')
                    ),
                })

                if wandb_run is not None:
                    # Deliberately no val/mean_* metrics: each QP is logged
                    # independently, which avoids redundant single-QP charts.
                    vlog = {'epoch': epoch}
                    for p in per_qp:
                        vlog[f"val/qp{p['qp']:02d}/bpp"] = p['bpp']
                        vlog[f"val/qp{p['qp']:02d}/psnr"] = p['psnr']
                        vlog[f"val/qp{p['qp']:02d}/psnr_y"] = p['psnr_y']
                        vlog[f"val/qp{p['qp']:02d}/psnr_u"] = p['psnr_u']
                        vlog[f"val/qp{p['qp']:02d}/psnr_v"] = p['psnr_v']
                        vlog[f"val/qp{p['qp']:02d}/num_images"] = p['num_images']
                        vlog[f"val/qp{p['qp']:02d}/skipped_oom"] = p['skipped_oom']
                        vlog[f"val/qp{p['qp']:02d}/skipped_nonfinite"] = (
                            p['skipped_nonfinite']
                        )
                    for view in ('current', 'all_epochs'):
                        for key, name in (
                            ('psnr_y', 'y'),
                            ('psnr_u', 'u'),
                            ('psnr_v', 'v'),
                        ):
                            if key in plot_paths[view]:
                                image = _wandb_image(wandb_run, plot_paths[view][key])
                                if image is not None:
                                    vlog[f'val-rd/{view}_{name}'] = image
                    validation_log = vlog

                if mean_psnr > best_psnr:
                    best_saved = _atomic_torch_save({
                        'state_dict': net_eval.state_dict(),
                        'epoch': epoch,
                        'val_mean_psnr': mean_psnr,
                        'validation_identity': rd_history_identity,
                    }, best_path)
                    if best_saved:
                        best_psnr = mean_psnr
                        best_epoch = epoch
                        _update_wandb_summary(wandb_run, {
                            'best/epoch': epoch,
                            'best/checkpoint_path': best_path,
                        })

            monitor_state_saved = _save_json(monitor_state_path, {
                'epoch': epoch,
                'global_step': global_step,
                'best_epoch': best_epoch,
                'best_psnr': best_psnr if math.isfinite(best_psnr) else None,
                'best_checkpoint_path': best_path if best_epoch is not None else None,
                'validation_identity': rd_history_identity,
            })
            save_status(args.save_dir, i_net, optimizer, epoch, verbose=False,
                        keep_num=args.keep_status_num)
            status_path = os.path.join(args.save_dir, f'status_epo{epoch}.pth.tar')
            _update_wandb_summary(wandb_run, {
                'checkpoint/latest_epoch': epoch,
                'checkpoint/latest_status_path': status_path,
                'checkpoint/monitor_state_saved': monitor_state_saved,
            })

            # Weights-only archive: a permanent record of this epoch that the
            # rolling status pruning never touches (loadable via get_state_dict).
            do_archive = args.archive_interval > 0 and (
                (epoch + 1) % args.archive_interval == 0 or epoch == args.epochs - 1)
            if do_archive:
                archive_dir = os.path.join(args.save_dir, 'archives')
                try:
                    os.makedirs(archive_dir, exist_ok=True)
                except OSError as e:
                    LOGGER.warning('failed to create archive directory %s: %s', archive_dir, e)
                archive_path = os.path.join(archive_dir, f'ckpt_epo{epoch:03d}.pth.tar')
                archive_saved = _atomic_torch_save({
                    'state_dict': net_eval.state_dict(),
                    'epoch': epoch,
                }, archive_path)
                _update_wandb_summary(wandb_run, {
                    'checkpoint/latest_archive_path': archive_path if archive_saved else '',
                    'checkpoint/latest_archive_epoch': epoch if archive_saved else None,
                })
            # Commit one W&B history row per epoch. Validation keys are merged
            # into the training row on validation epochs, avoiding duplicate
            # rows with the same custom epoch step.
            combined_log = {'epoch': epoch}
            if epoch_log is not None:
                combined_log.update(epoch_log)
            if validation_log is not None:
                combined_log.update(validation_log)
            if len(combined_log) > 1:
                _wandb_log(wandb_run, combined_log)

        if rank >= 0:
            dist.barrier()

    if rank <= 0:
        save_ckpt(args.save_dir, i_net, verbose=False)
        final_path = os.path.join(args.save_dir, 'ckpt.pth.tar')
        _update_wandb_summary(wandb_run, {
            'checkpoint/final_path': final_path,
            'checkpoint/completed_epochs': args.epochs,
        })
    if rank >= 0:
        dist.barrier()
    cleanup_train(rank)
    if rank <= 0:
        _wandb_finish(wandb_run)


def main(argv):
    args = parse_args(argv)
    validate_args(args)
    if args.deterministic:
        # cuBLAS reads this when its handle is created, i.e. before any CUDA work.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    start_train(train, args)


if __name__ == '__main__':
    main(sys.argv[1:])
