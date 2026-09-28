"""Bounded uniform-depth context check; not a mixed-map or runtime benchmark.

Two fixed DIV2K sources (0801, 0880), centre768 crops, QP32, e15 CPU FP32.
For each reachable depth, compare full-frame suffix features with independent
core32 windows carrying k-4 feature cells of context. Windows are clipped at
the true feature-image boundary; stock zero padding therefore occurs at the
correct global boundary on every layer. No outside-image feature band is
invented. The shared adapter/head is evaluated after assembly, without repair.
Primary tolerance is atol=rtol=1e-4 on features and unclipped444 outputs.
This numerical check does not claim bitwise equality or trained quality gains.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
SNAPSHOT = BASE / 'shared_metric_audit/source'
UPSTREAM = Path('/home/can_karsal/DCVC')
OUT = REPO / 'docs/research/2026-09-28-paper-editorial/uniform_context'
NAMES = ('0801.png', '0880.png')
DEPTHS = (6, 8, 10, 12)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, record):
    path = OUT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def compare(actual, reference):
    delta = actual - reference
    return {'max_abs': float(delta.abs().max()), 'rms': float(delta.square().mean().sqrt()),
            'reference_max_abs': float(reference.abs().max()),
            'within_declared_tolerance': bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4))}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    cp = REPO / 'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
    crop_manifest = BASE / 'div2k100_center512_rgb/manifest.json'
    image_records = {r['image']: r for r in json.loads(crop_manifest.read_text())['images']}
    original = json.loads((BASE / 'shared_crossfit_qp32/manifest.json').read_text())
    sources = {**original['source_sha256'], **{str(p): sha(p) for p in (Path(__file__), cp, crop_manifest)}}
    for name in NAMES:
        row = image_records[name]
        sources[row['source_path']] = row['source_sha256']
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    upstream = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if upstream != '819c219b24db34310bbd15c51a720aaaf5eb2e7d':
        raise ValueError('Upstream changed')
    manifest = {'scope': __doc__, 'images': list(NAMES), 'depths': list(DEPTHS),
                'crop': 'centre768, no resize, RGB to centred YCbCr444', 'qp': 32,
                'source_sha256': sources, 'upstream_commit': upstream, 'started_utc': utc(),
                'torch': torch.__version__, 'threads': 2}
    write('manifest.json', manifest)
    write('progress.json', {'state': 'running', 'pid': os.getpid(), 'completed': 0, 'total': 8})
    sys.path.insert(0, str(SNAPSHOT))
    sys.path.insert(0, str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import rgb2ycbcr_np
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ck['config'])
    assert cfg.split_depth == 2 and cfg.blocks_per_exit == 2 and cfg.feature_patch == 32
    net = FlexUFIntra(cfg).eval()
    net.load_state_dict(ck.get('state_dict', ck.get('net', ck)), strict=True)
    dec = net.dec
    rows = []
    try:
        with torch.inference_mode():
            for name in NAMES:
                with Image.open(image_records[name]['source_path']) as image:
                    rgb = np.array(image.convert('RGB'))
                h, w = rgb.shape[:2]
                assert min(h, w) >= 768
                rgb = rgb[(h-768)//2:(h+768)//2, (w-768)//2:(w+768)//2].copy()
                x = torch.from_numpy(rgb2ycbcr_np(rgb.astype(np.float32)/255.)-.5).permute(2,0,1)[None].contiguous()
                y, q, _ = net._encode_to_latent(x, torch.tensor([32], dtype=torch.int32))
                stem = dec.upsample(y)
                for group in dec.groups[:2]:
                    stem = group(stem)
                assert stem.shape[-2:] == (96, 96)
                for depth in DEPTHS:
                    stop = depth // 2
                    full = stem
                    for group in dec.groups[2:stop]:
                        full = group(full)
                    halo = depth - 4
                    assembled = torch.empty_like(full)
                    zero = torch.empty_like(full)
                    windows = []
                    for top in range(0, 96, 32):
                        for left in range(0, 96, 32):
                            t, l = max(0, top-halo), max(0, left-halo)
                            b, r = min(96, top+32+halo), min(96, left+32+halo)
                            feat = stem[:, :, t:b, l:r].contiguous()
                            for group in dec.groups[2:stop]:
                                feat = group(feat)
                            assembled[:, :, top:top+32, left:left+32] = feat[:, :, top-t:top-t+32, left-l:left-l+32]
                            windows.append([t, l, b, r])
                            # Boundary-control diagnostic with no overlap and stock zero padding.
                            feat = stem[:, :, top:top+32, left:left+32].contiguous()
                            for group in dec.groups[2:stop]:
                                feat = group(feat)
                            zero[:, :, top:top+32, left:left+32] = feat
                    full_image = dec._apply_head(dec._at_exit(full, stop-1), q)
                    tile_image = dec._apply_head(dec._at_exit(assembled, stop-1), q)
                    zero_image = dec._apply_head(dec._at_exit(zero, stop-1), q)
                    feature_check, image_check = compare(assembled, full), compare(tile_image, full_image)
                    row = {'image': name, 'depth': depth, 'feature_halo': halo,
                           'crop_pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                           'windows': windows, 'feature_check': feature_check, 'output444_check': image_check,
                           'zero_overlap_stock_padding_feature_difference': compare(zero, full),
                           'zero_overlap_stock_padding_output_difference': compare(zero_image, full_image),
                           'zero_overlap_control_scope': 'No overlap, stock zero padding, no repair. This is not the deployed replicate-padding+repair path.'}
                    rows.append(row)
                    write('results.json', {'manifest': manifest, 'rows': rows, 'state': 'running'})
                    if not (feature_check['within_declared_tolerance'] and image_check['within_declared_tolerance']):
                        raise AssertionError('Context reconstruction tolerance failed')
                    write('progress.json', {'state': 'running', 'pid': os.getpid(), 'completed': len(rows),
                                            'total': 8, 'image': name, 'depth': depth, 'updated_utc': utc()})
                    print(json.dumps({'image': name, 'depth': depth, 'feature': feature_check, 'output': image_check}), flush=True)
        for path, digest in sources.items():
            if sha(path) != digest:
                raise ValueError('Frozen source changed during check')
        if torch.cuda.is_initialized():
            raise AssertionError('Unexpected CUDA context')
        write('results.json', {'manifest': manifest, 'rows': rows, 'state': 'complete', 'finished_utc': utc(), 'cuda_initialized': False})
        write('progress.json', {'state': 'complete', 'completed': len(rows), 'total': 8, 'finished_utc': utc()})
    except BaseException as error:
        write('progress.json', {'state': 'failed', 'completed': len(rows), 'total': 8, 'error': repr(error), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
