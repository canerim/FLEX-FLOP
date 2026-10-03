"""Full Kodak actual-byte RD for final D2/D4/D6 and released D12.

CPU FP32 research rANS wire format with isolated bitstream-only decoder.
Resumable case records; never substitutes a neural entropy estimate for bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from PIL import Image
import torch

from model_io import ROOT as TRAIN, file_sha, load_model
from reference_codec import ReferenceCodec
sys.path.insert(0, str(TRAIN/'upstream'))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QPS = (0, 16, 32, 48, 63)
DEPTHS = (2, 4, 6, 12)
EXTENSION = TRAIN / 'research/reference_entropy_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def checkpoint(depth):
    return TRAIN / ('reference_d12/released_cvpr2026_image.pth.tar' if depth == 12
                    else f'runs/d{depth}/ckpt.pth.tar')


def receive(worker):
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f'Decoder process exited: {worker.poll()}')
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'results/kodak_final_verified')
    parser.add_argument('--max-cases', type=int, default=None, help='Smoke test only; never publish as full BD-rate')
    args = parser.parse_args()
    torch.set_num_threads(1)
    images = sorted((REPO/'data/kodak').glob('kodim*.png'))
    if len(images) != 24:
        raise RuntimeError(f'Expected 24 Kodak images, got {len(images)}')
    files = [HERE/name for name in ('evaluate.py', 'model_io.py', 'reference_codec.py', 'decode_worker.py')]
    manifest = {'schema': 1, 'dataset': 'Kodak 24 full-resolution original RGB PNG',
                'qps': QPS, 'depths': DEPTHS, 'cases': 24*5*4,
                'rate': 'actual FUFREF2 rANS payload bytes; container bytes also stored',
                'wire_scope': 'CPU FP32 research stream, NOT Microsoft native CUDA bitstream',
                'padding': 'image to multiple of 16; hyperlatent to multiple of 4',
                'metric': 'decoded YCbCr and clipped RGB on unpadded source pixels',
                'source_sha256': {p.name: file_sha(p) for p in images},
                'checkpoint_sha256': {str(d): file_sha(checkpoint(d)) for d in DEPTHS},
                'code_sha256': {p.name: file_sha(p) for p in files},
                'extension_manifest_sha256': file_sha(EXTENSION/'build_manifest.json'),
                'upstream_commit': 'cbdae87a5445114cdc7f48816da63ea80bdeac40',
                'torch': torch.__version__, 'device': 'cpu', 'threads_per_process': 1}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError('Existing manifest differs; use another output directory')
    atomic(manifest_path, manifest)
    from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb
    progress = {'state': 'running', 'completed': 0, 'total': manifest['cases'], 'started_utc': utc(), 'pid': os.getpid()}
    atomic(args.out/'progress.json', progress)
    completed_this_call = 0
    for depth in DEPTHS:
        folder = args.out/f'd{depth}'
        (folder/'cases').mkdir(parents=True, exist_ok=True)
        (folder/'streams').mkdir(exist_ok=True)
        net, info = load_model(checkpoint(depth), depth)
        codec = ReferenceCodec(net, info['sha256'], EXTENSION)
        with (folder/'decoder.stderr.log').open('a') as err:
            worker = subprocess.Popen([sys.executable, str(HERE/'decode_worker.py'),
                                       '--checkpoint', str(checkpoint(depth)), '--depth', str(depth)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=err, text=True, bufsize=1)
            try:
                ready = receive(worker)
                if not ready['ready'] or ready['checkpoint']['sha256'] != info['sha256']:
                    raise RuntimeError('Isolated decoder did not load the same checkpoint')
                for image in images:
                    raw = np.asarray(Image.open(image).convert('RGB'))
                    h, w = raw.shape[:2]
                    target = torch.from_numpy(raw.astype(np.float32)/255).permute(2,0,1)[None]
                    x = torch.from_numpy(rgb2ycbcr_np(raw.astype(np.float32)/255)-.5).permute(2,0,1)[None].contiguous()
                    for qp in QPS:
                        name = f'{image.stem}_qp{qp:02d}'
                        case_path = folder/'cases'/f'{name}.json'
                        stream_path = folder/'streams'/f'{name}.fufref2'
                        if case_path.exists():
                            row = json.loads(case_path.read_text())
                            if (row['stream_sha256'] != file_sha(stream_path) or
                                    row['checkpoint_sha256'] != info['sha256'] or
                                    row['source_sha256'] != manifest['source_sha256'][image.name] or
                                    not row['exact_isolated_decode']):
                                raise RuntimeError(f'Existing case integrity fails: {case_path}')
                        else:
                            encoded = codec.encode(x, qp)
                            stream_path.write_bytes(encoded.stream)
                            decoded_path = folder/f'{name}_decoded.npy'
                            worker.stdin.write(json.dumps({'stream': str(stream_path), 'output': str(decoded_path)})+'\n')
                            worker.stdin.flush()
                            response = receive(worker)
                            if not response.get('ok') or not response.get('source_analysis_disabled'):
                                raise RuntimeError('Isolated decoder failed')
                            decoded = np.load(decoded_path, allow_pickle=False)
                            decoded_path.unlink()
                            if not np.array_equal(decoded, encoded.reconstruction.numpy()):
                                raise RuntimeError(f'Encode/decode reconstruction mismatch: {name}')
                            diag = response['diagnostics']
                            for field in ('z_hat_sha256', 'y_hat_sha256'):
                                if diag[field] != encoded.diagnostics[field]:
                                    raise RuntimeError(f'{field} differs: {name}')
                            for left, right in zip(diag['stages'], encoded.diagnostics['stages']):
                                for field in ('symbols', 'symbols_sha256', 'indexes_sha256'):
                                    if left[field] != right[field]:
                                        raise RuntimeError(f'Entropy trace mismatch: {name}/{field}')
                            rec = torch.from_numpy(decoded).clamp(-.5, .5)
                            rgb = ycbcr2rgb(rec+.5, clamp=True)
                            rgb_mse = float((rgb-target).square().mean())
                            yuv_mse = (rec-x).square().mean(dim=(0,2,3)).tolist()
                            psnr = [-10*math.log10(max(v, 1e-12)) for v in yuv_mse]
                            row = {'image': image.name, 'qp': qp, 'depth': depth, 'h': h, 'w': w,
                                   'source_sha256': manifest['source_sha256'][image.name],
                                   'checkpoint_sha256': info['sha256'],
                                   'stream_sha256': file_sha(stream_path),
                                   'payload_bytes': encoded.diagnostics['payload_bytes'],
                                   'container_bytes': len(encoded.stream),
                                   'payload_bpp': encoded.diagnostics['payload_bpp'],
                                   'container_bpp': encoded.diagnostics['container_bpp'],
                                   'estimated_bpp_diagnostic_only': encoded.diagnostics['estimated_bpp'],
                                   'psnr_rgb': -10*math.log10(max(rgb_mse, 1e-12)),
                                   'psnr_yuv611': (6*psnr[0]+psnr[1]+psnr[2])/8,
                                   'psnr_y': psnr[0], 'psnr_u': psnr[1], 'psnr_v': psnr[2],
                                   'exact_isolated_decode': True,
                                   'entropy_trace': diag['stages'], 'finished_utc': utc()}
                            atomic(case_path, row)
                        progress['completed'] += 1
                        progress.update(depth=depth, image=image.name, qp=qp, updated_utc=utc())
                        atomic(args.out/'progress.json', progress)
                        print(json.dumps({'type': 'case', 'completed': progress['completed'],
                                          'total': progress['total'], 'depth': depth,
                                          'image': image.name, 'qp': qp, 'payload_bpp': row['payload_bpp']}), flush=True)
                        completed_this_call += 1
                        if args.max_cases and completed_this_call >= args.max_cases:
                            progress.update(state='partial_smoke', updated_utc=utc())
                            atomic(args.out/'progress.json', progress)
                            return
            finally:
                if worker.poll() is None:
                    worker.stdin.write('{"stop":true}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=30)
    if progress['completed'] != progress['total']:
        raise RuntimeError('Incomplete cohort')
    progress.update(state='complete', finished_utc=utc(), cuda_initialized=False)
    atomic(args.out/'progress.json', progress)


if __name__ == '__main__':
    main()
