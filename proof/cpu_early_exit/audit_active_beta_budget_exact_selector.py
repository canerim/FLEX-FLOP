"""Check every frozen budget price against exact convolution MAC ranking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent/'results/div2k_beta/quality_floor'
POLICY = BASE/'active_replicate_budget_sweep_policy.json'
EXACT = BASE/'active_beta_exact_selector_audit.json'
OUT = BASE/'active_beta_budget_exact_selector_audit.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    policy = json.loads(POLICY.read_text())
    exact = json.loads(EXACT.read_text())
    records = []
    for target in policy['targets']:
        for qp in policy['qps']:
            candidates = [r for r in exact['per_qp'][str(qp)]['candidates']
                          if r['mean_delta444_db'] <= target+1e-12]
            assert candidates
            optimum = max(candidates,
                          key=lambda r: (r['exact_mean_saving_pct'],
                                         -r['mean_delta444_db'], -r['beta']))
            chosen = policy['policies'][str(target)]['selected'][str(qp)]
            selected = next(r for r in candidates if r['beta'] == chosen['beta'])
            assert abs(chosen['mean_delta444_db']-selected['mean_delta444_db'])<1e-12
            assert abs(chosen['mean_conv_mac_saving_pct']-
                       selected['legacy_mean_saving_pct'])<1e-12
            records.append({'target_db':target, 'qp':qp,
                            'selected_beta':selected['beta'],
                            'exact_optimum_beta':optimum['beta'],
                            'same_beta':selected['beta']==optimum['beta'],
                            'selected_exact_conv_mac_saving_pct':
                            selected['exact_mean_saving_pct']})
    output={'scope':'Read-only calibration candidate audit; no validation outcome or policy revision.',
            'source_sha256':{'policy':sha(POLICY),'exact_candidate_audit':sha(EXACT),
                             'script':sha(Path(__file__))},
            'all_match':all(r['same_beta'] for r in records),
            'comparisons':records}
    assert len(records)==20 and output['all_match']
    OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'comparisons':len(records),'all_match':output['all_match']},indent=2))


if __name__=='__main__':
    main()
