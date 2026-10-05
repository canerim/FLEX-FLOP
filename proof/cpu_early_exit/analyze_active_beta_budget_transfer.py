"""Analyze calibration-locked quality-budget transfer without retuning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from exact_active_conv_mac import exact_active_no_repair_saving_pct

HERE = Path(__file__).resolve().parent
BASE = HERE / 'results/div2k_beta/quality_floor'
RAW = BASE / 'active_replicate_budget_transfer_validation24.json'
POLICY = BASE / 'active_replicate_budget_sweep_policy.json'
OUT = BASE / 'active_replicate_budget_transfer_analysis.json'
QPS = (0, 16, 32, 48, 63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows: list[dict], fn, rng: np.random.Generator) -> dict:
    images = sorted({r['image'] for r in rows})
    n_qp = len(rows)//len(images)
    assert len(images) == 24 and n_qp in (1, 5)
    mat = np.asarray([[fn(r) for r in rows if r['image']==image]
                      for image in images], dtype=float)
    assert mat.shape == (24, n_qp)
    boot = mat[rng.integers(0, 24, size=(10000, 24))].mean(axis=(1, 2))
    return {'mean': float(mat.mean()),
            'image_cluster_ci95': np.quantile(boot,[.025,.975]).tolist(),
            'min': float(mat.min()), 'max': float(mat.max())}


def main() -> None:
    raw = json.loads(RAW.read_text())
    policy = json.loads(POLICY.read_text())
    assert raw['complete'] and len(raw['rows']) == raw['expected_cases'] == 120
    assert raw['provenance']['policy_sha256'] == sha(POLICY)
    assert raw['targets'] == policy['targets']
    assert raw['feasible_targets'] == [str(t) for t in policy['targets']
                                     if policy['policies'][str(t)]['complete_five_qp_policy']]
    rows = [{**r, 'targets': {target: {**record,
              'exact_conv_mac_saving_pct':exact_active_no_repair_saving_pct(record['exit_map'])}
              for target,record in r['targets'].items()}} for r in raw['rows']]
    keys = {(r['image'],r['qp']) for r in rows}
    assert len(keys) == 120 and len({im for im,_ in keys}) == 24
    assert all({qp for im,qp in keys if im==image} == set(QPS)
               for image in {im for im,_ in keys})
    rng = np.random.default_rng(20261009)
    policies = {}
    for target in raw['feasible_targets']:
        assert all(set(r['targets']) == set(raw['feasible_targets']) for r in rows)
        assert all(r['targets'][target]['beta'] == policy['policies'][target]['beta'][str(r['qp'])]
                   for r in rows)
        loss = lambda r: r['targets'][target]['delta444_db']
        saving = lambda r: r['targets'][target]['conv_mac_saving_pct']
        exact_saving = lambda r: r['targets'][target]['exact_conv_mac_saving_pct']
        qps = {}
        for qp in QPS:
            subset = [r for r in rows if r['qp']==qp]
            qps[str(qp)] = {
                'mean_delta444_db': summary(subset, loss, rng),
                'mean_conv_mac_saving_pct': summary(subset, saving, rng),
                'mean_exact_conv_mac_saving_pct':summary(subset,exact_saving,rng),
                'over_0p1_count': sum(loss(r) > .1 for r in subset),
                'validation_mean_within_calibration_target':
                    sum(loss(r) for r in subset)/24 <= float(target),
            }
        policies[target] = {
            'mean_delta444_db': summary(rows, loss, rng),
            'mean_conv_mac_saving_pct': summary(rows, saving, rng),
            'mean_exact_conv_mac_saving_pct':summary(rows,exact_saving,rng),
            'over_0p1_count': sum(loss(r) > .1 for r in rows),
            'per_qp': qps,
        }
        if target != '0.1':
            policies[target]['saving_change_vs_0p1_points'] = summary(
                rows, lambda r: saving(r)-r['targets']['0.1']['conv_mac_saving_pct'], rng)
            policies[target]['exact_saving_change_vs_0p1_points'] = summary(
                rows, lambda r: exact_saving(r)-r['targets']['0.1']['exact_conv_mac_saving_pct'], rng)
            policies[target]['quality_change_vs_0p1_db'] = summary(
                rows, lambda r: r['targets']['0.1']['delta444_db']-loss(r), rng)
    output = {
        'scope': 'Exploratory validation transfer of predeclared calibration-only quality budgets; frozen e15/FUFREF2, untimed analytical synthesis-conv MAC.',
        'source_sha256': {'raw': sha(RAW), 'policy': sha(POLICY),
                          'script': sha(Path(__file__)),
                          'exact_conv_formula':sha(HERE/'exact_active_conv_mac.py')},
        'bootstrap': '10000 image-cluster draws retaining five QPs per image',
        'all_targets': raw['targets'], 'feasible_targets': raw['feasible_targets'],
        'policies': policies,
    }
    OUT.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'feasible_targets': output['feasible_targets'],
                      'policies': {target: {'mean_loss': p['mean_delta444_db']['mean'],
                                           'mean_exact_saving': p['mean_exact_conv_mac_saving_pct']['mean'],
                                           'over_0p1': p['over_0p1_count']}
                                   for target,p in policies.items()}}, indent=2))


if __name__ == '__main__':
    main()
