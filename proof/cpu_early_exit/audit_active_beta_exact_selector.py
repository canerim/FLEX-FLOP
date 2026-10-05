"""Check frozen beta selection under exact active-context convolution MAC."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exact_active_conv_mac import exact_active_no_repair_saving_pct

HERE=Path(__file__).resolve().parent
BASE=HERE/'results/div2k_beta'
FRONTIER=BASE/'quality_floor/active_replicate_beta_calibration24.json'
MANIFEST=BASE/'manifest.json'
POLICY=BASE/'quality_floor/active_replicate_beta_locked_policy.json'
VALIDATION=BASE/'quality_floor/active_replicate_beta_validation24.json'
OUT=BASE/'quality_floor/active_beta_exact_selector_audit.json'
QPS=(0,16,32,48,63)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    frontier=json.loads(FRONTIER.read_text())
    manifest=json.loads(MANIFEST.read_text())
    policy=json.loads(POLICY.read_text())
    validation=json.loads(VALIDATION.read_text())
    assert frontier['complete'] and len(frontier['rows'])==120
    assert validation['complete'] and len(validation['rows'])==120
    assert frontier['provenance']['manifest_sha256']==sha(MANIFEST)
    assert policy['source_sha256']['frontier']==sha(FRONTIER)
    assert validation['provenance']['policy_sha256']==sha(POLICY)
    by_key={(r['image'],r['qp']):r for r in frontier['rows']}
    assert len(by_key)==120
    calibration_images=sorted({image for image,_ in by_key})
    assert len(calibration_images)==24
    tables={}
    for qp in QPS:
        betas=manifest['candidates'][str(qp)]
        table=[]
        for index,beta in enumerate(betas):
            cases=[]
            for image in calibration_images:
                row=by_key[image,qp]
                candidate=row['candidates'][index]
                assert candidate['beta']==beta
                map_row=row['maps'][candidate['map_id']]
                cases.append((map_row['active_delta444_db'],
                              map_row['active_no_repair_conv_saving_pct'],
                              exact_active_no_repair_saving_pct(map_row['exit_map'])))
            loss=sum(c[0] for c in cases)/24
            table.append({'beta':beta,'mean_delta444_db':loss,
                          'legacy_mean_saving_pct':sum(c[1] for c in cases)/24,
                          'exact_mean_saving_pct':sum(c[2] for c in cases)/24,
                          'eligible':loss<=.1})
        eligible=[r for r in table if r['eligible']]
        assert eligible
        exact_winner=max(eligible,key=lambda r:(r['exact_mean_saving_pct'],
                                                 -r['mean_delta444_db'],-r['beta']))
        legacy=max(eligible,key=lambda r:(r['legacy_mean_saving_pct'],
                                           -r['mean_delta444_db'],-r['beta']))
        selected=[r for r in table if r['beta']==policy['beta'][str(qp)]]
        assert len(selected)==1 and selected[0]['beta']==legacy['beta']
        tables[str(qp)]={'selected':selected[0],'exact_optimum':exact_winner,
                         'same_beta':exact_winner['beta']==selected[0]['beta'],
                         'candidates':table}
    validation_rows=[]
    for row in validation['rows']:
        exact=exact_active_no_repair_saving_pct(row['exit_map'])
        validation_rows.append({'image':row['image'],'qp':row['qp'],
                                'selected_beta':row['selected_beta'],
                                'exact_saving_pct':exact,
                                'legacy_saving_pct':row['new_conv_mac_saving_pct'],
                                'delta444_db':row['new_delta444_db']})
    result={
        'scope':'Post-lock exact-convolution audit of calibration-only active-replicate/no-repair beta. Does not revise the frozen policy or use validation to select beta.',
        'source_sha256':{'frontier':sha(FRONTIER),'manifest':sha(MANIFEST),
                         'policy':sha(POLICY),'validation':sha(VALIDATION),
                         'formula':sha(HERE/'exact_active_conv_mac.py'),
                         'script':sha(Path(__file__))},
        'same_beta_at_all_qps':all(r['same_beta'] for r in tables.values()),
        'per_qp':tables,
        'validation_mean_exact_saving_pct':sum(r['exact_saving_pct'] for r in validation_rows)/120,
        'validation_mean_legacy_saving_pct':sum(r['legacy_saving_pct'] for r in validation_rows)/120,
        'validation_rows':validation_rows,
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'same_beta_at_all_qps':result['same_beta_at_all_qps'],
                      'betas':{q:{'locked':r['selected']['beta'],
                                   'exact_optimum':r['exact_optimum']['beta']}
                               for q,r in tables.items()},
                      'validation_mean_exact_saving_pct':result['validation_mean_exact_saving_pct']},indent=2))


if __name__=='__main__':main()
