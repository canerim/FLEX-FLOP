"""Strict four-model Kodak BD-rate from actual rANS payload bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from reference_codec import parse_container

DEPTHS = (2, 4, 6, 12)
QPS = (0, 16, 32, 48, 63)
METRICS = ('psnr_rgb', 'psnr_yuv611')


def curve(rows, metric, rate):
    ordered = sorted(rows, key=lambda r: r[metric])
    q = np.array([r[metric] for r in ordered])
    bpp = np.array([r[rate] for r in ordered])
    if np.any(np.diff(q) <= 0) or np.any(np.diff(bpp) <= 0) or np.any(bpp <= 0):
        raise ValueError('Non-monotone or nonpositive curve')
    return q, PchipInterpolator(q, np.log(bpp), extrapolate=False)


def analyze(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    progress = json.loads((folder/'progress.json').read_text())
    if progress.get('state') != 'complete' or progress.get('completed') != 480:
        raise RuntimeError('All 480 actual-bitstream cases must complete before BD-rate')
    if tuple(manifest['depths']) != DEPTHS or tuple(manifest['qps']) != QPS:
        raise RuntimeError('Cohort manifest differs')
    by = {}
    for depth in DEPTHS:
        rows = [json.loads(p.read_text()) for p in sorted((folder/f'd{depth}'/'cases').glob('*.json'))]
        if len(rows) != 120 or len({(r['image'],r['qp']) for r in rows}) != 120:
            raise RuntimeError(f'D{depth} has missing or duplicate cases')
        if any(r['depth'] != depth or not r['exact_isolated_decode'] for r in rows):
            raise RuntimeError(f'D{depth} decode verification failed')
        for row in rows:
            if (row['source_sha256'] != manifest['source_sha256'][row['image']] or
                    row['checkpoint_sha256'] != manifest['checkpoint_sha256'][str(depth)]):
                raise RuntimeError('Source or checkpoint identity differs from frozen manifest')
            stream_path = folder/f'd{depth}'/'streams'/f'{Path(row["image"]).stem}_qp{row["qp"]:02d}.fufref2'
            stream = stream_path.read_bytes()
            if hashlib.sha256(stream).hexdigest() != row['stream_sha256']:
                raise RuntimeError(f'Stream hash differs: {stream_path}')
            parsed = parse_container(stream, depth, row['checkpoint_sha256'])
            if (parsed['height'], parsed['width'], parsed['qp']) != (row['h'], row['w'], row['qp']):
                raise RuntimeError(f'Stream metadata differs: {stream_path}')
            if (parsed['payload_bytes'] != row['payload_bytes'] or len(stream) != row['container_bytes'] or
                    not math.isclose(row['payload_bpp'], 8*parsed['payload_bytes']/(row['h']*row['w']), rel_tol=1e-12)):
                raise RuntimeError(f'Rate does not match emitted bytes: {stream_path}')
        by[depth] = {image: [r for r in rows if r['image'] == image]
                     for image in sorted({r['image'] for r in rows})}
        if len(by[depth]) != 24 or any({r['qp'] for r in curve_rows} != set(QPS)
                                        for curve_rows in by[depth].values()):
            raise RuntimeError(f'D{depth} lacks complete images/QPs')
    images = sorted(by[12])
    if any(sorted(by[depth]) != images for depth in DEPTHS):
        raise RuntimeError('Image identities differ between codecs')
    rng = np.random.default_rng(20261003)
    output = {'scope': 'Kodak24 x five QPs x D2/D4/D6/released D12; exact isolated decode',
              'wire_format': manifest['wire_scope'], 'rate_definition': 'actual emitted rANS payload bytes',
              'method': 'per-image PCHIP integration in log-rate on each image four-model common PSNR support; arithmetic mean of image BD-rates; no extrapolation',
              'metrics': {}, 'qp_means': {}}
    for depth in DEPTHS:
        output['qp_means'][f'D{depth}'] = {str(qp): {
            'payload_bpp': float(np.mean([r['payload_bpp'] for rs in by[depth].values() for r in rs if r['qp'] == qp])),
            'psnr_rgb': float(np.mean([r['psnr_rgb'] for rs in by[depth].values() for r in rs if r['qp'] == qp])),
            'psnr_yuv611': float(np.mean([r['psnr_yuv611'] for rs in by[depth].values() for r in rs if r['qp'] == qp]))
        } for qp in QPS}
    for metric in METRICS:
        per = {d: {} for d in DEPTHS if d != 12}
        supports = {}
        for image in images:
            cs = {depth: curve(by[depth][image], metric, 'payload_bpp') for depth in DEPTHS}
            low = max(c[0][0] for c in cs.values())
            high = min(c[0][-1] for c in cs.values())
            if high <= low:
                raise RuntimeError(f'No shared support for {image}/{metric}')
            supports[image] = [float(low), float(high)]
            for depth in per:
                delta = (cs[depth][1].integrate(low, high)-cs[12][1].integrate(low, high))/(high-low)
                per[depth][image] = 100*math.expm1(delta)
        output['metrics'][metric] = {'common_psnr_support_db': supports, 'bd_rate_vs_released_d12': {}}
        for depth, cases in per.items():
            vals = np.array([cases[image] for image in images])
            draws = vals[rng.integers(0, len(vals), (5000, len(vals)))].mean(axis=1)
            output['metrics'][metric]['bd_rate_vs_released_d12'][f'D{depth}'] = {
                'mean_pct': float(vals.mean()), 'image_bootstrap_ci95_pct': [float(v) for v in np.quantile(draws, (.025,.975))],
                'n_images': len(vals), 'per_image_pct': cases}
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, default=Path(__file__).parent/'results/kodak_final_verified')
    a = p.parse_args()
    result = analyze(a.folder)
    path = a.folder/'analysis.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({metric: {name: row['mean_pct'] for name,row in details['bd_rate_vs_released_d12'].items()}
                      for metric,details in result['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
