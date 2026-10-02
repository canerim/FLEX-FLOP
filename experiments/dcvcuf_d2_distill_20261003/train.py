"""D2 output distillation from Microsoft's released DCVC-UF-Intra image model.

The student retains the official RD objective, data loader, optimizer, batch size,
QP sampling, 105-epoch LR/patch schedule and FP32 settings. Intentional changes:
D2 warm start and a weighted teacher-output MSE term. Teacher is frozen.
"""
import argparse
import datetime
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import numpy as np
sys.path.insert(0, '/data10/shareddata/can_karsal/dcvcuf_depth_20260927/code_snapshot_v2')
from runtime import (ROOT, COMMIT, UPSTREAM, ImageFolder, atomic_json, check_upstream,
                     file_sha, get_dataloader, get_training_lambdas, get_training_strategy,
                     init_train, model_for_depth, recipe, restore_rng, rng_state)
from runtime import compile_preserving_rng
from src.models.image_model import DMCI
from src.utils.common import loss_func

STOP = False
TEACHER_PATH = ROOT / 'reference_d12/released_cvpr2026_image.pth.tar'
STUDENT_PATH = ROOT / 'runs/d2/ckpt.pth.tar'
ALPHA = 0.25


class StudentRD(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, x, qp):
        return self.student.forward_one_frame(x, qp)


def stop_requested(signum, frame):
    global STOP
    STOP = True


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_state(folder, net, optimizer, epoch, next_batch, epoch_rng, global_step,
               manifest, skipped, elapsed):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict(),
             'epoch': epoch, 'next_batch': next_batch, 'epoch_start_rng': epoch_rng,
             'rng': rng_state(), 'global_step': global_step, 'manifest': manifest,
             'nonfinite_skips': skipped, 'elapsed_training_seconds': elapsed}
    tmp = folder / 'resume.tmp.pt'
    torch.save(state, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    current = folder / 'resume.pt'
    previous = folder / 'resume.previous.pt'
    if current.exists():
        os.replace(current, previous)
    os.replace(tmp, current)


def load_state(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck.get('state_dict', ck.get('net', ck))


def distillation_loss(result, teacher_output, lambdas):
    # Both outputs use Microsoft's centered YCbCr convention. The original RD
    # objective is unchanged; this is an explicit auxiliary term per sample.
    official = loss_func(result, lambdas)['loss']
    per_image = (result['x_hat'] - teacher_output).square().mean(dim=(1, 2, 3))
    auxiliary = (lambdas * per_image).mean()
    return official + ALPHA * auxiliary, official.detach(), auxiliary.detach()


def update_distilled(student, runner, teacher, optimizer, batch):
    x, qp, lambdas = batch
    with torch.no_grad():
        # Limit teacher activation memory in the official 512px stage; there
        # are no batch-dependent layers in this eval-mode teacher.
        micro = 4 if x.shape[-1] > 256 else 16
        target = torch.cat([
            teacher.forward_one_frame(x[j:j+micro], qp[j:j+micro], recon_only=True)
            for j in range(0, x.shape[0], micro)
        ], dim=0)
    result = runner(x, qp)
    loss, official, auxiliary = distillation_loss(result, target, lambdas)
    optimizer.zero_grad()
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.1,
                                               error_if_nonfinite=False).item()
    finite = np.isfinite(total_norm)
    if finite:
        optimizer.step()
    return loss.detach(), float(official), float(auxiliary), float(total_norm), bool(finite)


def train(args):
    check_upstream()
    folder = Path(args.save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1, 'One visible GPU per independent experiment required'
    init_train(-1, str(folder))
    if args.deterministic_check:
        assert args.smoke, 'Deterministic mode is a verification-only control, not the production recipe'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    data_root = Path(args.train_dataset)
    if not args.smoke:
        ready = json.loads((data_root / 'READY.json').read_text())
        assert ready['status'] == 'ready' and ready['done'] == 384795
        assert ready['description_sha256'] == file_sha(data_root / 'description.json')
    ds = ImageFolder(str(data_root), 256, 256, 64, get_training_lambdas([10, 2048], 64))
    if not args.smoke:
        assert len(ds) == 384795
    assert args.depth == 2
    assert TEACHER_PATH.is_file() and STUDENT_PATH.is_file()
    net = model_for_depth(2)
    net.load_state_dict(load_state(STUDENT_PATH), strict=True)
    net.cuda()
    teacher = DMCI()
    teacher.load_state_dict(load_state(TEACHER_PATH), strict=True)
    teacher.cuda().eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    runner = StudentRD(net)
    if args.compile:
        runner = compile_preserving_rng(net)
    manifest = {**recipe(), 'depth': args.depth, 'seed': 42,
                'official_reference_recipe': recipe(),
                'loss': 'unmodified upstream RD loss + auxiliary released-output MSE',
                'initialization': 'warm start from completed D2 epoch 105; released D12 teacher is frozen',
                'torch': torch.__version__, 'python': sys.version,
                'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'compile': args.compile, 'memory_format': 'contiguous',
                'compile_preserves_pytorch_rng': True,
                'num_images': len(ds), 'description_sha256': file_sha(data_root / 'description.json'),
                'parameters': sum(p.numel() for p in net.parameters()),
                'decoder_parameters': sum(p.numel() for p in net.dec.parameters()),
                'all_parameters_trainable': all(p.requires_grad for p in net.parameters()),
                'training_files_sha256': {'train.py': file_sha(Path(__file__)),
                                          'runtime.py': file_sha(ROOT/'code_snapshot_v2/runtime.py'),
                                          'evaluate.py': file_sha(ROOT/'code_snapshot_v2/evaluate.py')},
                'upstream_files_sha256': {name: file_sha(UPSTREAM / name) for name in
                                         ['train_image.py', 'src/models/image_model.py',
                                          'src/layers/layers.py', 'src/datasets/image_dataset.py',
                                          'src/utils/common.py']}, 'smoke': args.smoke,
                'student_initial_checkpoint': str(STUDENT_PATH),
                'student_initial_sha256': file_sha(STUDENT_PATH),
                'teacher_checkpoint': str(TEACHER_PATH),
                'teacher_sha256': file_sha(TEACHER_PATH),
                'teacher_mode': 'eval, frozen, same augmented input and QP',
                'teacher_microbatch': '16 at 256px; 4 at 512px',
                'distillation': 'original RD + 0.25 * mean(lambda_QP * MSE_YCbCr(student_recon, released_teacher_recon))',
                'distillation_alpha': ALPHA,
                'warm_start': 'final 105-epoch D2; optimizer reset to official AdamW defaults',
                'dataset_scope': 'all 384795 images in official Open Images train_0/1/2'}
    state = None
    for resume in [folder / 'resume.pt', folder / 'resume.previous.pt']:
        if resume.exists():
            try:
                # Only our own locally written checkpoints are loaded here.
                state = torch.load(resume, map_location='cpu', weights_only=False)
                for key in ['depth', 'upstream_commit', 'description_sha256', 'compile', 'smoke',
                            'training_files_sha256', 'torch', 'seed', 'student_initial_sha256',
                            'teacher_sha256', 'distillation_alpha']:
                    assert state['manifest'][key] == manifest[key], (key, resume)
                net.load_state_dict(state['net'], strict=True)
                optimizer.load_state_dict(state['opt'])
                print(f'Resuming {resume} epoch={state["epoch"]} next_batch={state["next_batch"]}', flush=True)
                break
            except (EOFError, RuntimeError, OSError) as exc:
                print(f'Could not load {resume}: {exc}', flush=True)
                state = None
    atomic_json(folder / 'manifest.json', manifest)
    global_step = state['global_step'] if state else 0
    skipped = state['nonfinite_skips'] if state else 0
    elapsed_before = state.get('elapsed_training_seconds', 0.) if state else 0.
    begin_epoch = state['epoch'] if state else 0
    start = time.monotonic()
    strategy = get_training_strategy()
    events = open(folder / 'metrics.jsonl', 'a', buffering=1)
    row = {'timestamp': timestamp(), 'pid': os.getpid(), 'depth': args.depth,
           'epoch': begin_epoch, 'global_step': global_step, 'nonfinite_skips': skipped,
           'state': 'initializing'}
    next_batch = 0
    # A fresh loader each epoch is also how upstream non-persistent workers
    # pick up set_patch_size at epoch 90.
    for epoch in range(begin_epoch, 105):
        _, lr, width, height = strategy[epoch]
        if args.smoke and args.smoke_patch:
            width = height = args.smoke_patch
        ds.set_patch_size(width, height)
        for group in optimizer.param_groups:
            group['lr'] = lr
        loader = get_dataloader(ds, -1, 1, 16, 4)
        resume_batch = state['next_batch'] if state and epoch == begin_epoch else 0
        if state and epoch == begin_epoch:
            restore_rng(state['epoch_start_rng'])
        epoch_rng = rng_state()
        iterator = iter(loader)
        # Replaying worker batches reconstructs sampler, augmentations and QPs
        # exactly. Only IO/augmentation is replayed, never optimizer updates.
        for _ in range(resume_batch):
            next(iterator)
        if state and epoch == begin_epoch:
            restore_rng(state['rng'])
            state = None
        net.train()
        teacher.eval()
        window_start = time.monotonic()
        window_count = 0
        losses = []
        for i, batch in enumerate(iterator, start=resume_batch):
            batch = [t.to('cuda') for t in batch]
            loss, rd_loss, teacher_mse_scaled, norm, finite = update_distilled(
                net, runner, teacher, optimizer, batch)
            skipped += int(not finite)
            global_step += 1
            next_batch = i + 1
            window_count += 1
            if not finite:
                print(f'non-finite norm; official recipe skips batch epoch={epoch} i={i}', flush=True)
            if i % 200 == 0 or STOP or args.smoke and global_step >= args.smoke_steps:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - window_start
                row = {'timestamp': timestamp(), 'pid': os.getpid(), 'depth': args.depth,
                       'epoch': epoch, 'next_batch': next_batch, 'batches_per_epoch': len(loader),
                       'global_step': global_step, 'optimizer_updates': global_step - skipped,
                       'lr': lr, 'patch': width, 'loss': float(loss) if torch.isfinite(loss) else None,
                       'grad_norm': norm if math.isfinite(norm) else None, 'nonfinite_skips': skipped,
                       'seconds_per_step_window': elapsed / window_count,
                       'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                       'elapsed_training_seconds': elapsed_before + time.monotonic() - start,
                       'state': 'running', 'rd_loss': rd_loss,
                       'teacher_mse_lambda_scaled': teacher_mse_scaled,
                       'distillation_alpha': ALPHA}
                events.write(json.dumps(row) + '\n')
                atomic_json(folder / 'status.json', row)
                print(json.dumps(row), flush=True)
                window_start, window_count = time.monotonic(), 0
            if global_step % args.checkpoint_steps == 0 or STOP or args.smoke and global_step >= args.smoke_steps:
                save_state(folder, net, optimizer, epoch, next_batch, epoch_rng, global_step,
                           manifest, skipped, elapsed_before + time.monotonic() - start)
            if global_step == args.checkpoint_steps and not args.smoke and not STOP:
                from evaluate import validation
                try:
                    validation(net, folder, global_step, epoch)
                except Exception as exc:
                    print(f'Validation failed (training continues): {exc!r}', flush=True)
                    atomic_json(folder/'validation_failure.json', {'error': repr(exc), 'global_step': global_step})
                window_start, window_count = time.monotonic(), 0
            if STOP or args.smoke and global_step >= args.smoke_steps:
                row['state'] = 'stopped' if STOP else 'smoke_complete'
                atomic_json(folder / 'status.json', row)
                events.close()
                return
        # Capture the RNG for starting the following epoch before constructing
        # its iterator, so resuming at an epoch boundary also reproduces data.
        epoch_rng = rng_state()
        save_state(folder, net, optimizer, epoch + 1, 0, epoch_rng, global_step,
                   manifest, skipped, elapsed_before + time.monotonic() - start)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save({'state_dict': net.state_dict(), 'depth': args.depth, 'epoch': epoch + 1,
                        'manifest': manifest}, folder / f'weights_epoch{epoch + 1:03d}.pt')
        if not args.smoke:
            from evaluate import validation
            try:
                validation(net, folder, global_step, epoch+1)
            except Exception as exc:
                print(f'Validation failed (training continues): {exc!r}', flush=True)
                atomic_json(folder/'validation_failure.json', {'error': repr(exc), 'global_step': global_step})
        next_batch = 0
    torch.save({'state_dict': net.state_dict(), 'depth': args.depth, 'epoch': 105,
                'manifest': manifest}, folder / 'ckpt.pth.tar')
    row['state'] = 'complete'
    row['epoch'] = 105
    row['next_batch'] = 0
    row['global_step'] = global_step
    row['optimizer_updates'] = global_step - skipped
    row['nonfinite_skips'] = skipped
    row['elapsed_training_seconds'] = elapsed_before + time.monotonic() - start
    row['timestamp'] = timestamp()
    atomic_json(folder / 'status.json', row)
    events.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, choices=[2], default=2)
    ap.add_argument('--train-dataset', default=str(ROOT / 'openimages_012_png'))
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--checkpoint-steps', type=int, default=2000)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-steps', type=int, default=10)
    ap.add_argument('--smoke-patch', type=int, choices=[256, 512])
    ap.add_argument('--deterministic-check', action='store_true')
    args = ap.parse_args()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop_requested)
    try:
        train(args)
    except BaseException as exc:
        folder = Path(args.save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'failure.json', {'timestamp': timestamp(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
