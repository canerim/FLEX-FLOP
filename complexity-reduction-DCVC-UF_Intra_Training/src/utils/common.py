# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import timedelta
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def cleanup_train(rank):
    if rank >= 0:
        dist.destroy_process_group()


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f'created folder: {path}')


def generate_str(x):
    return '  '.join(f'{a.item():.5f}' for a in x) + '  '


def get_current_device(rank, verbose=True):
    if verbose and rank <= 0:
        print(f'cuda device count: {torch.cuda.device_count()}')

    if rank >= 0:
        device = f'cuda:{rank}'
    elif torch.cuda.device_count() > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'
    if verbose:
        print(f'rank: {rank}, current device: {device}')
    return device


def get_dataloader(dataset, rank, world_size, batch_size, num_workers, generator=None,
                   seed=0):
    """Build the training loader.

    ``generator`` drives both the sampling order and the per-worker seeds
    (workers derive theirs from the loader's base seed), so passing a seeded
    generator makes an epoch's data order and augmentation reproducible.
    """
    if rank >= 0:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                           seed=seed)
        assert batch_size % world_size == 0
        arg_batch_size = batch_size // world_size
    else:
        train_sampler = RandomSampler(dataset, generator=generator)
        arg_batch_size = batch_size
    return DataLoader(
        dataset,
        batch_size=arg_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=2,
        generator=generator,
        # Workers survive across epochs: with a large num_workers the respawn
        # (and its dataset re-pickling) happens once instead of every epoch.
        persistent_workers=num_workers > 0,
    )


def get_latest_status_path(dir_cur):
    files = os.listdir(dir_cur)
    all_status_files = [os.path.join(dir_cur, f) for f in files if 'status_epo' in f]
    all_status_files.sort(key=os.path.getmtime)
    if len(all_status_files) > 2:
        return all_status_files[-2:]
    return all_status_files


def loss_func(rd, lambdas):
    costs = lambdas * rd['mse'] + rd['bpp']
    return {
        'losses': costs,
        'loss': torch.mean(costs),
    }


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if 'net' in ckpt:
        ckpt = ckpt['net']
    consume_prefix_in_state_dict_if_present(ckpt, prefix='module.')
    return ckpt


def get_training_lambdas(lambdas, qp_num):
    all_lambdas = np.linspace(np.log(lambdas[0]), np.log(lambdas[1]), qp_num)
    all_lambdas = np.exp(all_lambdas)
    return all_lambdas


def init_train(rank, save_dir, timeout_minutes=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world_size = 1
    if rank >= 0:
        torch.cuda.set_device(rank)
        world_size = torch.cuda.device_count()
        timeout_kwargs = (
            {'timeout': timedelta(minutes=timeout_minutes)}
            if timeout_minutes is not None else {}
        )
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank, **timeout_kwargs)

    if rank <= 0:
        create_folder(save_dir)
    if rank >= 0:
        # Rank 0 owns directory creation. Keep the remaining ranks from
        # inspecting the checkpoint directory before it exists.
        dist.barrier()

    return world_size


def load_existing_weights(save_dir, net, rank, pretrain_path=None, verbose=True):
    begin_epoch = 0
    opt_status = None
    ckpt_loaded = False
    existing_status_path = get_latest_status_path(save_dir)
    for status_path in reversed(existing_status_path):
        try:
            status = torch.load(status_path, map_location=torch.device('cpu'),
                                weights_only=True)
            opt_status = status['opt']
            begin_epoch = status['epoch'] + 1
            ckpt_loaded = True
            if rank <= 0:
                net_status = status['net']
                consume_prefix_in_state_dict_if_present(net_status, prefix='module.')
                net.load_state_dict(net_status)
                if verbose:
                    print(f'load status from {status_path}')
                    print(f'begin epoch {begin_epoch}')
            break
        except Exception:
            continue

    if not ckpt_loaded and pretrain_path is not None:
        if rank <= 0:
            net_state_dict = get_state_dict(pretrain_path)
            net.load_state_dict(net_state_dict)
            if verbose:
                print(f'load pretrained weights from {pretrain_path}')

    return begin_epoch, opt_status


def save_ckpt(save_dir, net, verbose=True):
    ckpt_path = os.path.join(save_dir, 'ckpt.pth.tar')
    net = net.module if isinstance(net, DDP) else net
    torch.save({'state_dict': net.state_dict()}, ckpt_path)
    if verbose:
        print(f'save final checkpoint to {ckpt_path}')


def save_status(save_dir, net, opt, epoch, verbose=True, keep_num=1):
    """Write the resume point for ``epoch`` and prune older ones.

    The write goes to a temporary file and is renamed into place, so an
    interrupted save can never leave a truncated resume point behind. Set
    ``keep_num`` above 1 to retain older status files -- ``get_latest_status_path``
    hands the two newest to ``load_existing_weights``, which falls back to the
    older one when the newest fails to load.
    """
    curr_path = os.path.join(save_dir, f'status_epo{epoch}.pth.tar')
    net = net.module if isinstance(net, DDP) else net
    save_dict = {
        'epoch': epoch,
        'net': net.state_dict(),
        'opt': opt.state_dict(),
    }
    # The temporary name deliberately avoids the 'status_epo' substring so a
    # leftover partial file is never picked up as a resume candidate.
    fd, tmp_path = tempfile.mkstemp(dir=save_dir, prefix=f'.tmp_stat{epoch}.', suffix='.tmp')
    os.close(fd)
    try:
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, curr_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    if verbose:
        print(f'save model epoch {epoch}')

    status_files = [
        os.path.join(save_dir, f) for f in os.listdir(save_dir) if 'status_epo' in f
    ]
    status_files = [f for f in status_files if os.path.isfile(f)]
    status_files.sort(key=os.path.getmtime)
    for full_path in status_files[:-max(int(keep_num), 1)]:
        if full_path != curr_path and os.path.exists(full_path):
            os.remove(full_path)


def start_train(train_fun, args):
    if torch.cuda.device_count() > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355)
        world_size = torch.cuda.device_count()
        mp.spawn(train_fun, nprocs=world_size, args=(args,), join=True)
    else:
        train_fun(-1, args)


def wrap_ddp(net, rank, find_unused_parameters=False):
    if rank >= 0:
        net = DDP(net, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    return net

