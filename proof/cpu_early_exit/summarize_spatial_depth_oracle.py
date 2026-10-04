"""Summarize the source-informed +1 tile-subset ablation without model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


DEFAULT = Path(__file__).resolve().parent / 'results/div2k_beta/quality_floor'


def summarize(base: dict, spatial: dict, uniform: dict, uniform_rows: dict) -> dict:
    cases = base['rows']
    failures = spatial['rows']
    if len(cases) != 120 or len(failures) != 24:
        raise ValueError('Expected 120 fixed cases and 24 exact-context failures')
    keys = {(r['image'], r['qp']) for r in cases}
    failure_keys = {(r['image'], r['qp']) for r in failures}
    if len(keys) != 120 or len(failure_keys) != 24 or not failure_keys <= keys:
        raise ValueError('Missing or duplicate image/QP cases')
    counts = Counter(r['minimum_upgraded_tiles'] for r in failures)
    rescued = [r for r in failures if r['best_subset'] is not None]
    uniform_by_key = {(r['image'], r['qp']): r for r in uniform_rows['rows']}
    final_saving = {}
    for r in failures:
        key = (r['image'], r['qp'])
        if r['best_subset'] is not None:
            assert r['best_subset']['exact_context_delta444_db'] <= .1
            final_saving[key] = r['best_subset']['ideal_shared_saving_pct']
        else:
            assert uniform_by_key[key]['chosen']['exact_context_delta444_db'] <= .1
            final_saving[key] = uniform_by_key[key]['chosen']['ideal_shared_saving_pct']
    cohort = [final_saving.get((r['image'], r['qp']), r['ideal_shared_saving_pct'])
              for r in cases]
    spatial_mean = mean(cohort)
    uniform_mean = uniform['uniform_increment_on_failures_mean_ideal_saving_pct']
    return {
        'schema': 1,
        'scope': 'Source-informed exact-context spatial +1 subset diagnostic; ideal synthesis MAC only',
        'cases': 120, 'failures': 24,
        'minimum_upgraded_tiles_distribution': {str(k): counts[k] for k in (1, 2, 3, 4, 5, 6, None)},
        'spatial_subset_feasible_failures': len(rescued),
        'spatial_feasible_failure_mean_ideal_saving_pct': mean(r['best_subset']['ideal_shared_saving_pct'] for r in rescued),
        'spatial_fallback_full_cohort_mean_ideal_saving_pct': spatial_mean,
        'uniform_fallback_full_cohort_mean_ideal_saving_pct': uniform_mean,
        'baseline_full_cohort_mean_ideal_saving_pct': uniform['base_mean_ideal_saving_pct'],
        'spatial_vs_uniform_full_cohort_pct_points': spatial_mean - uniform_mean,
        'source_feedback_required': True,
        'latency_measured': False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=DEFAULT)
    d = p.parse_args().directory
    result = summarize(
        json.loads((d/'exact_context_validation24.json').read_text()),
        json.loads((d/'single_tile_oracle_validation24.json').read_text()),
        json.loads((d/'uniform_depth_oracle_summary.json').read_text()),
        json.loads((d/'uniform_depth_oracle_validation24.json').read_text()),
    )
    (d/'spatial_depth_oracle_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
