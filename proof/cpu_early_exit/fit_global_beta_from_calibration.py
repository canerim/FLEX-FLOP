"""Fit an exploratory *global mean* beta budget using calibration only.

This second rule was specified after inspecting calibration aggregates, so it
must be labelled exploratory. Neither validation nor Kodak records are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--per-qp-policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = json.loads(args.per_qp_policy.read_text())
    if source['manifest_sha256'] != sha(args.manifest):
        raise RuntimeError('Calibration table belongs to a different manifest')
    qps = [str(q) for q in manifest['qps']]
    if set(source['calibration_tables']) != set(qps):
        raise RuntimeError('Incomplete calibration tables')

    # Exact Pareto pruning: discard a state only when another has no worse
    # mean quality and at least as much MAC saving at this stage.
    frontier = [(0.0, 0.0, [])]
    for qp in qps:
        candidates = source['calibration_tables'][qp]
        if [r['beta'] for r in candidates] != manifest['candidates'][qp]:
            raise RuntimeError(f'Calibration candidate grid differs at QP{qp}')
        combined = sorted(((delta + r['mean_delta444_db'],
                            saving + r['mean_mac_saved_pct'], betas + [r['beta']])
                           for delta, saving, betas in frontier for r in candidates),
                          key=lambda row: (row[0], -row[1]))
        frontier = []
        max_saving = float('-inf')
        for state in combined:
            if state[1] > max_saving + 1e-9:
                frontier.append(state)
                max_saving = state[1]
    target = float(manifest['target_delta444_db_mean'])
    feasible = [r for r in frontier if r[0] / len(qps) <= target]
    if not feasible:
        raise RuntimeError('No feasible global policy on locked candidate grid')
    best = max(feasible, key=lambda r: (r[1], -r[0], tuple(-v for v in r[2])))
    policy = {'schema': 1,
              'scope': 'Locked beta chosen on DIV2K calibration images only; exploratory global-QP mean budget rule fixed before Kodak transfer',
              'manifest_sha256': sha(args.manifest),
              'calibration_source_policy_sha256': sha(args.per_qp_policy),
              'selection_rule': 'Maximize equal-QP mean analytical synthesis MAC saving with equal-QP mean Delta444 <= 0.10 dB on 24 DIV2K calibration images; Pareto search of the same locked candidate grid',
              'status': 'exploratory rule chosen after inspecting calibration aggregate; validation and Kodak not consulted',
              'beta': dict(zip(qps, best[2])),
              'calibration_mean_delta444_db': best[0] / len(qps),
              'calibration_mean_mac_saved_pct': best[1] / len(qps),
              'calibration_result_sha256': source['calibration_result_sha256']}
    if args.output.exists() and json.loads(args.output.read_text()) != policy:
        raise RuntimeError('A different global policy is already locked')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2) + '\n')
    print(json.dumps({'policy_sha256': sha(args.output), 'beta': policy['beta'],
                      'calibration_mean_delta444_db': policy['calibration_mean_delta444_db'],
                      'calibration_mean_mac_saved_pct': policy['calibration_mean_mac_saved_pct']}, indent=2))


if __name__ == '__main__':
    main()
