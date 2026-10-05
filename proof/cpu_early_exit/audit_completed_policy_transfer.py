"""Cross-file integrity audit for completed frozen early-exit transfers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from exact_active_conv_mac import exact_active_no_repair_saving_pct


HERE = Path(__file__).resolve().parent
BASE = HERE/'results/div2k_beta/quality_floor'
CLIC = HERE/'results/clic39_active_beta'
OUT = BASE/'completed_policy_transfer_integrity.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    values = {(r['image'],r['qp']):r for r in rows}
    assert len(values)==len(rows)
    return values


def close(a: float,b: float) -> bool:
    return abs(a-b)<1e-9


def main() -> None:
    files = {
        'primary_raw': BASE/'active_replicate_beta_validation24.json',
        'budget_raw': BASE/'active_replicate_budget_transfer_validation24.json',
        'budget_analysis': BASE/'active_replicate_budget_transfer_analysis.json',
        'tail_raw': BASE/'active_replicate_tail_validation24.json',
        'tail_analysis': BASE/'active_replicate_tail_validation24_analysis.json',
        'bayer_raw': BASE/'active_beta_bayer_validation24.json',
        'bayer_analysis': BASE/'active_beta_bayer_validation24_analysis.json',
        'clic_raw': CLIC/'raw.json',
        'clic_summary': CLIC/'summary.json',
    }
    d={name:read(path) for name,path in files.items()}
    for name in ('primary_raw','budget_raw','tail_raw','bayer_raw'):
        assert d[name]['complete'] and len(d[name]['rows'])==120
    p=index(d['primary_raw']['rows'])
    b=index(d['budget_raw']['rows'])
    t=index(d['tail_raw']['rows'])
    y=index(d['bayer_raw']['rows'])
    assert set(p)==set(b)==set(t)==set(y)
    assert {q for _,q in p}=={0,16,32,48,63}
    for key,primary in p.items():
        budget,tail,bayer=b[key],t[key],y[key]
        assert primary['stream_sha256']==budget['stream_sha256']==tail['stream_sha256']==bayer['stream_sha256']
        assert primary['case_sha256']==budget['case_sha256']==tail['case_sha256']==bayer['case_sha256']
        midpoint=budget['targets']['0.1']
        assert midpoint['exit_map']==primary['exit_map']==tail['primary_exit_map']==bayer['router_map']
        assert close(midpoint['delta444_db'],primary['new_delta444_db'])
        assert close(tail['primary_delta444_db'],primary['new_delta444_db'])
        assert close(bayer['router_delta444_db'],primary['new_delta444_db'])
        assert sorted(bayer['router_map'])==sorted(bayer['bayer_map'])
        assert close(bayer['placement_gain_db'],
                     10*math.log10(bayer['bayer_mse444']/bayer['router_mse444']))
        assert close(exact_active_no_repair_saving_pct(primary['exit_map']),
                     exact_active_no_repair_saving_pct(bayer['bayer_map']))
    assert d['budget_analysis']['source_sha256']['raw']==sha(files['budget_raw'])
    assert d['tail_analysis']['source_sha256']['raw']==sha(files['tail_raw'])
    assert d['bayer_analysis']['source_sha256']['raw']==sha(files['bayer_raw'])
    assert d['budget_analysis']['policies']['0.1']['over_0p1_count']==34
    assert d['tail_analysis']['thresholds']['tail_over_0p1']==8
    assert close(sum(x['placement_gain_db'] for x in y.values())/120,
                 d['bayer_analysis']['overall']['mean_db'])
    clic=d['clic_raw']
    assert clic['complete'] and len(clic['rows'])==195
    c=index(clic['rows'])
    assert len({im for im,_ in c})==39 and {q for _,q in c}=={0,16,32,48,63}
    assert d['clic_summary']['input_sha256']==sha(files['clic_raw'])
    assert d['clic_summary']['n_images']==39 and d['clic_summary']['n_cases']==195
    assert all(v['available']['n']==39 for v in d['clic_summary']['bd_rate_summary'].values())
    result={
        'scope':'Read-only cross-file integrity; not independent model evaluation or native codec reproduction.',
        'source_sha256':{name:sha(path) for name,path in files.items()},
        'script_sha256':sha(Path(__file__)),
        'matched_div2k_cases':120,'matched_clic_cases':195,
        'same_stream_and_case_hash_across_div2k_arms':True,
        'budget_0p1_equals_primary_route_and_quality':True,
        'bayer_histograms_and_exact_macs_match_router':True,
        'clic_all_bd_rate_curves_available':True,
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
