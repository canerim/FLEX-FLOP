"""Run the predeclared Kodak3x3 paired bitstream or full-roundtrip cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from bitstream_benchmark import (DEFAULT_EXTENSION, DEFAULT_RELEASE,
                                 DEFAULT_UPSTREAM, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=HERE/'results/bitstream_kodak3x3/manifest.json')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--include-encoder', action='store_true')
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--extension', type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument('--out', type=Path,
                        default=HERE/'results/bitstream_cohort')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('cases') != 9 or manifest.get('images') !=
            ['kodim01.png', 'kodim13.png', 'kodim24.png'] or
            manifest.get('qps') != [16, 32, 48]):
        raise RuntimeError('Unexpected or incomplete predeclared cohort')
    if args.blocks < 5:
        parser.error('Need at least five paired blocks per case')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in manifest['rows']:
        stream = REPO/row['stream']
        source = REPO/'data/kodak'/row['image']
        if digest(stream) != row['stream_sha256'] or digest(source) != row['image_sha256']:
            raise RuntimeError(f'Predeclared input changed: {stream}')
        out = args.out/f"{Path(row['image']).stem}_qp{row['qp']}.json"
        reuse = False
        if out.exists():
            old = json.loads(out.read_text())
            reuse = bool(old.get('claim_eligible') and
                         old.get('include_encoder') == args.include_encoder and
                         old.get('stream_sha256') == row['stream_sha256'] and
                         old.get('source_sha256') == row['image_sha256'] and
                         old.get('gpu_index') == args.gpu and
                         old.get('blocks') == args.blocks and
                         old.get('warmup') == args.warmup and
                         old.get('extension_manifest_sha256') ==
                         digest(args.extension/'build_manifest.json') and
                         old.get('checkpoint_sha256', {}).get('released') ==
                         digest(args.release) and
                         old.get('code_sha256', {}).get('benchmark') ==
                         digest(HERE/'bitstream_benchmark.py') and
                         old.get('code_sha256', {}).get('planned_decoder') ==
                         digest(REPO/'flexuf/kernels/planned_decoder.py'))
        if not reuse:
            command = [sys.executable, str(HERE/'bitstream_benchmark.py'),
                       '--upstream', str(args.upstream), '--release', str(args.release),
                       '--extension', str(args.extension), 'benchmark',
                       '--stream', str(stream), '--source', str(source),
                       '--gpu', str(args.gpu), '--blocks', str(args.blocks),
                       '--warmup', str(args.warmup), '--out', str(out)]
            if args.include_encoder:
                command.append('--include-encoder')
            print(json.dumps({'case': out.name, 'event': 'start'}), flush=True)
            subprocess.run(command, check=True)
        result = json.loads(out.read_text())
        if not result['claim_eligible'] or not result['quality_yuv611_444_db']:
            raise RuntimeError(f'Incomplete speed/quality proof: {out}')
        results.append({
            'image': row['image'], 'qp': row['qp'],
            'result': str(out.relative_to(REPO)),
            'stream_sha256': row['stream_sha256'],
            'stream_bytes': row['stream_bytes'],
            'median_ms': result['median_ms'],
            'speedup_stock_pair_median':
                statistics.median(result['paired_speedups']['stock_vs_stock']),
            'speedup_matched_triton_pair_median':
                statistics.median(result['paired_speedups']['matched_triton']),
            'released_minus_e15_yuv611_444_db':
                result['released_minus_e15_yuv611_444_db'],
            'route_counts': result['route_counts'],
        })
        print(json.dumps({'case': out.name, 'event': 'complete',
                          'matched_triton_speedup':
                          results[-1]['speedup_matched_triton_pair_median']}), flush=True)
    values = [r['speedup_matched_triton_pair_median'] for r in results]
    summary = {
        'schema': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'claim_eligible': True, 'scope': ('full CPU research image->bytes->GPU image'
                                         if args.include_encoder else
                                         'CPU research bytes->GPU image'),
        'manifest_sha256': digest(args.manifest), 'cases': len(results),
        'gpu_index': args.gpu, 'blocks_per_case': args.blocks,
        'median_of_case_medians_matched_triton': statistics.median(values),
        'observed_case_median_range_matched_triton': [min(values), max(values)],
        'rows': results,
        'inference_limit': 'Three predeclared Kodak images times three QPs; range is observed workload variation, not a population confidence interval',
    }
    target = args.out/'cohort_summary.json'
    target.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summary': str(target),
                      'median_speedup': summary['median_of_case_medians_matched_triton']}))


if __name__ == '__main__':
    main()
