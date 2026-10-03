"""Recompute a nine-scenario released/e15 speed claim from raw idle-GPU trials."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

SEQUENCES = ('videoSRC05', 'FourPeople', 'BQMall')
QPS = (16, 32, 48)
ARMS = ('released_d12', 'e15_stock', 'e15_triton')


def summarize(paths):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    if len(rows) != 9:
        raise ValueError('Nine independently recorded scenario runs required')
    keys = {(r['sequence'].split('_')[0], r['qp']) for r in rows}
    if keys != {(seq, qp) for seq in SEQUENCES for qp in QPS}:
        raise ValueError('Expected 3 specified sequences x QP 16/32/48, once each')
    if len({tuple(sorted(r['checkpoint_sha256'].items())) for r in rows}) != 1:
        raise ValueError('Checkpoint identities vary between runs')
    for field in ('benchmark_sha256', 'archived_maps_sha256', 'torch'):
        if len({r[field] for r in rows}) != 1:
            raise ValueError(f'{field} varies between runs')
    if len({r['device'] for r in rows}) != 1:
        raise ValueError('Mixed GPU types')
    matched = ['released_d12_triton' in r['samples'] for r in rows]
    if any(matched) and not all(matched):
        raise ValueError('Cannot mix matched-kernel and stock-only runs')
    for sequence in SEQUENCES:
        group = [r for r in rows if r['sequence'].startswith(sequence)]
        for field in ('first_frame_sha256', 'source_shape', 'padded_shape'):
            if len({json.dumps(r[field], sort_keys=True) for r in group}) != 1:
                raise ValueError(f'{field} differs across {sequence} QPs')
    result = []
    for row in rows:
        if not row['claim_eligible'] or row['gpu_occupants_before'] or row['gpu_occupants_after'] or row['gpu_interference_during_blocks']:
            raise ValueError(f'Shared GPU in {row["sequence"]}/QP{row["qp"]}')
        if row['shared_tensors_exact'] != 255 or row['e15_stock_vs_triton_max_abs'] > 1e-4:
            raise ValueError('Model or optimized output check failed')
        samples = row['samples']
        count = len(samples['released_d12'])
        if count < 20 or any(len(samples[arm]) != count for arm in ARMS):
            raise ValueError('Missing paired repetitions')
        if all(matched):
            if (len(samples['released_d12_triton']) != count or
                    row['released_stock_vs_triton_max_abs'] is None or
                    row['released_stock_vs_triton_max_abs'] > 1e-4):
                raise ValueError('Missing or invalid matched released-D12 control')
        ratios = []
        for i in range(count):
            release = samples['released_d12'][i]['wall_ms']
            fast = samples['e15_triton'][i]['wall_ms']
            if not math.isfinite(release/fast) or min(release, fast) <= 0:
                raise ValueError('Invalid timing')
            ratios.append(release/fast)
        median = statistics.median(ratios)
        if not math.isclose(median, row['median_speedup_wall'], rel_tol=1e-9):
            raise ValueError('Saved speedup disagrees with raw samples')
        case = {'sequence': row['sequence'], 'qp': row['qp'],
                       'paired_median_speedup': median,
                       'released_median_ms': statistics.median(x['wall_ms'] for x in samples['released_d12']),
                       'e15_triton_median_ms': statistics.median(x['wall_ms'] for x in samples['e15_triton']),
                       'released_yuv611_db': row['quality_yuv611_db']['released_d12'],
                       'e15_yuv611_db': row['quality_yuv611_db']['e15_triton']}
        if all(matched):
            paired = [samples['released_d12_triton'][i]['wall_ms'] /
                      samples['e15_triton'][i]['wall_ms'] for i in range(count)]
            if any(not math.isfinite(x) or x <= 0 for x in paired):
                raise ValueError('Invalid matched-kernel timing')
            case['paired_median_speedup_matched_kernels'] = statistics.median(paired)
            case['released_triton_median_ms'] = statistics.median(
                x['wall_ms'] for x in samples['released_d12_triton'])
        result.append(case)
    medians = [r['paired_median_speedup'] for r in result]
    per_sequence = {sequence: statistics.median(r['paired_median_speedup'] for r in result
                                                if r['sequence'].startswith(sequence))
                    for sequence in SEQUENCES}
    summary = {'scope': 'decoder synthesis only, nine first-frame workloads, isolated GPU',
            'device': rows[0]['device'], 'checkpoint_sha256': rows[0]['checkpoint_sha256'],
            'median_of_scenario_paired_medians': statistics.median(medians),
            'min_scenario_paired_median': min(medians),
            'max_scenario_paired_median': max(medians),
            'per_sequence_median_of_qps': per_sequence,
            'uncertainty_note': 'Three source frames with correlated QPs; report workload range and raw paired samples, not a population confidence interval.',
            'cases': sorted(result, key=lambda r: (r['sequence'], r['qp']))}
    if all(matched):
        values = [r['paired_median_speedup_matched_kernels'] for r in result]
        summary['matched_kernels_median_of_scenario_paired_medians'] = statistics.median(values)
        summary['matched_kernels_scenario_range'] = [min(values), max(values)]
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', nargs=9, type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).parent/'results/cohort_summary.json')
    args = p.parse_args()
    summary = summarize(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
