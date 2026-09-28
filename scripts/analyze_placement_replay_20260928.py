"""Validate and summarise the complete, fixed-protocol spatial placement replay.

No partial-cohort estimates are produced. Bootstrap intervals condition on
the three recorded permutations; they do not estimate permutation-sampling
uncertainty, training variability or performance on an external test set.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
SOURCE = BASE / 'placement_replay_qp32'
ORIGINAL = BASE / 'shared_crossfit_qp32'
GROUPS = REPO / 'cvpr2027/data/research20260927/shared_crossfit_qp32/proposed_content_groups.json'
OUT = REPO / 'docs/research/2026-09-28-paper-editorial/placement_replay'
SEEDS = (202609280, 202609281, 202609282)
POLICIES = ('router', 'dither')
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_DRAWS = 5000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flat(value):
    if isinstance(value, list):
        return [item for child in value for item in flat(child)]
    require(isinstance(value, int) and 0 <= value < 6, 'Invalid exit index')
    return [value]


def describe(values, draws):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and np.isfinite(x).all(), 'Invalid observations')
    means = np.asarray([x[indices].mean() for indices in draws])
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max()),
            'ci95': np.quantile(means, [.025, .975]).tolist()}


def analyse(source, output):
    state = read(source / 'progress.json')
    require(state.get('state') == 'complete' and state.get('completed_sequences') == 53,
            'All 53 sequences must complete before any outcome analysis')
    require(state.get('cuda_initialized') is False, 'CPU-only completion check missing')
    manifest = read(source / 'manifest.json')
    require(manifest['permutation_seeds'] == list(SEEDS), 'Unexpected permutation seeds')
    require(manifest['policies'] == list(POLICIES), 'Unexpected policies')
    require(manifest['qp'] == 32 and manifest['criterion'] == 'q90', 'Wrong operating point')
    for path, digest in manifest['source_sha256'].items():
        require(sha(path) == digest, f'Changed experiment input: {path}')
    for path, digest in read(ORIGINAL / 'manifest.json')['source_sha256'].items():
        require(sha(path) == digest, f'Changed baseline source: {path}')
    paths = sorted((source / 'cases').glob('*.json'))
    cases = [read(path) for path in paths]
    names = sorted(manifest['names'])
    require(len(cases) == len(names) == len(set(names)) == 53, 'Wrong cohort size')
    require(Counter(c['sequence'] for c in cases) == Counter(names), 'Cohort identities differ')
    require(len({c['sequence_index'] for c in cases}) == 53, 'Repeated sequence index')
    records, invariants = [], []
    for case in sorted(cases, key=lambda c: c['sequence']):
        name, h, w = case['sequence'], case['height'], case['width']
        baseline = read(ORIGINAL / 'cases' / f"{case['sequence_index']:02d}.json")
        require((name, h, w, case['frame_sha256']) ==
                (baseline['sequence'], baseline['height'], baseline['width'],
                 baseline['first_frame_bytes_sha256']), 'Baseline identity differs')
        require(case['baseline_reproduced'] is True, 'Baseline reproduction failed')
        nx, ny = math.ceil(w / 256), math.ceil(h / 256)
        n = nx * ny
        strata = defaultdict(list)
        for i in range(n):
            row, column = divmod(i, nx)
            strata[(min(256, h - row * 256), min(256, w - column * 256))].append(i)
        expected_orders = []
        for seed in SEEDS:
            order = list(range(n))
            rng = random.Random(int(hashlib.sha256(f'{name}:{seed}'.encode()).hexdigest(), 16))
            for group in strata.values():
                shuffled = group.copy()
                rng.shuffle(shuffled)
                for destination, origin in zip(group, shuffled):
                    order[destination] = origin
            expected_orders.append(order)
        require(case['permutation_indices'] == expected_orders, 'Permutation protocol changed')
        rows = {(r['policy'], r['variant'], r['seed']): r for r in case['rows']}
        expected_keys = {(p, 'original', None) for p in POLICIES} | {
            (p, 'permuted', seed) for p in POLICIES for seed in SEEDS}
        require(len(case['rows']) == len(rows) == 8 and set(rows) == expected_keys,
                'Missing or repeated policy/seed case')
        maps_executed = set()
        for policy in POLICIES:
            original = rows[policy, 'original', None]
            original_map = flat(original['map'])
            old = next(r for r in baseline['rows'] if (r['criterion'], r['policy']) == ('q90', policy))
            require(original_map == flat(old['map']) and len(original_map) == n, 'Original map changed')
            for metric, old_key in [('rgb', 'cropped_rgb_mse'), ('444', 'cropped_ycbcr444_mse')]:
                require(abs(original[f'mse_{metric}'] - old[old_key]) <= 1e-12,
                        'Original reconstruction does not reproduce the archived replay')
            all_rows = [original] + [rows[policy, 'permuted', seed] for seed in SEEDS]
            for position, row in enumerate(all_rows):
                mapping = flat(row['map'])
                expected_map = original_map if position == 0 else [original_map[i] for i in expected_orders[position - 1]]
                require(mapping == expected_map, 'Recorded map disagrees with permutation')
                require(row['tile_histogram'] == [original_map.count(i) for i in range(6)], 'Histogram record differs')
                for group in strata.values():
                    require(Counter(mapping[i] for i in group) == Counter(original_map[i] for i in group),
                            'Valid-area stratum histogram changed')
                require(row['saving_points'] == old['saving_points'], 'Recorded compute accounting changed')
                for metric in ('rgb', '444'):
                    mse = row[f'mse_{metric}']
                    require(math.isfinite(mse) and mse > 0, 'Invalid MSE')
                    delta = 10 * math.log10(mse / original[f'mse_{metric}'])
                    require(abs(delta - row[f'{metric}_loss_vs_original_db']) < 1e-7, 'Stored loss does not match MSE')
                    if mapping == original_map:
                        require(mse == original[f'mse_{metric}'], 'Identical map produced different metric')
                maps_executed.add(tuple(mapping))
            record = {'sequence': name, 'policy': policy}
            for metric in ('rgb', '444'):
                shuffled = [r[f'mse_{metric}'] for r in all_rows[1:]]
                record[f'{metric}_placement_gain_db'] = 10 * math.log10(
                    float(np.mean(shuffled)) / original[f'mse_{metric}'])
                record[f'{metric}_mean_permutation_loss_db'] = float(np.mean([
                    10 * math.log10(m / original[f'mse_{metric}']) for m in shuffled]))
            records.append(record)
            invariants.append({'sequence': name, 'policy': policy,
                               'unchanged_permutations': sum(flat(r['map']) == original_map for r in all_rows[1:]),
                               'all_sampled_maps_identical': all(flat(r['map']) == original_map for r in all_rows)})
        require(case['distinct_maps_executed'] == len(maps_executed), 'Execution cache accounting differs')

    lookup = {(r['sequence'], r['policy']): r for r in records}
    metrics = [key for key in records[0] if key not in ('sequence', 'policy')]
    paired = [{'sequence': name, **{key: lookup[name, 'router'][key] - lookup[name, 'dither'][key]
                                  for key in metrics}} for name in names]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(names), size=(BOOTSTRAP_DRAWS, len(names)))
    groups = read(GROUPS)['groups']
    require(Counter(name for group in groups for name in group) == Counter(names), 'Content groups differ')
    cluster_rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_indices = [np.asarray([names.index(name) for name in group]) for group in groups]
    cluster_draws = [np.concatenate([group_indices[i] for i in selection])
                     for selection in cluster_rng.integers(0, len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))]
    summaries = {}
    for policy in (*POLICIES, 'router_minus_dither'):
        values = paired if policy == 'router_minus_dither' else [lookup[name, policy] for name in names]
        summaries[policy] = {metric: {
            'sequence_bootstrap': describe([r[metric] for r in values], draws),
            'content_cluster_sensitivity': describe([r[metric] for r in values], cluster_draws)} for metric in metrics}
    result = {'manifest': manifest, 'n_sequences': 53, 'n_map_cases': 424,
              'n_distinct_reconstructions': sum(c['distinct_maps_executed'] for c in cases),
              'per_sequence': records, 'paired_contrasts': paired, 'summaries': summaries,
              'invariant_maps': invariants,
              'statistics': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
                'weighting': 'Equal sequence weight, paired policies and all permutations kept together.',
                'cluster_sensitivity': 'Resample 51 pre-existing candidate-content groups, retaining their members; sequence-weighted point estimate unchanged. This sensitivity analysis does not change the fitted folds.',
                'interval_scope': 'Conditional on these three permutations, fixed Q90 controls, checkpoint and development-used cohort. Does not quantify permutation Monte Carlo error or independent generalisation.'},
              'provenance': {str(p): sha(p) for p in [Path(__file__), GROUPS, source / 'manifest.json', source / 'progress.json', *paths]}}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Sabit derinlik dağılımında gerçek yerleştirme kontrolü', '',
             '53 ilk frame × 2 sabit Q90 politika × (orijinal + 3 permütasyon) = 424 harita vakası. '
             'Tüm vakalar tamamlandı; orijinal haritalar önceki CPU replay çıktısını yeniden üretti. '
             'Tile derinlik sayıları ve geçerli piksel alanı tabakalarının derinlik dağılımları korundu.', '',
             'Birincil ölçüm her görüntü için 10 log10(üç permütasyonun ortalama RGB MSE’si / orijinal RGB MSE). '
             'Pozitif değer orijinal yerleştirmenin daha iyi olduğunu belirtir. Ardından 53 görüntünün eşit ağırlıklı ortalaması alınır.', '',
             '| Politika / fark | RGB yerleştirme kazancı (dB) | %95 sequence bootstrap aralığı |',
             '|---|---:|---:|']
    for policy, summary in summaries.items():
        stat = summary['rgb_placement_gain_db']['sequence_bootstrap']
        lo, hi = stat['ci95']
        lines.append(f"| {policy} | {stat['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'Aralıklar yalnız bu üç permütasyona koşulludur; permütasyon örnekleme hatasını, '
              'yeniden eğitim belirsizliğini veya bağımsız test genellemesini kapsamaz. '
              'Değişmeyen haritalar ve uniform örnekler çıkarılmadı. 51 içerik grubuyla duyarlılık analizi JSON içinde yer alır. '
              'Önceki tile-error-table permütasyon tanısıyla protokol ve metrik farklıdır; sayılar doğrudan birleştirilmemelidir. '
              'Bu deney bitstream veya hız ölçümü değildir.']
    (output / 'REPORT_TR.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'n_sequences': 53, 'n_map_cases': 424, 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    analyse(args.source, args.output)
