"""Exact whole-crop allocation diagnostics on fixed-rate interim RD curves.

Source-informed, interpolated quality and architectural MACs only. This is
not spatial routing within a crop, a trained MLP, or an observed speedup.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'docs/research/2026-09-27-six-hour'
DEPTHS=(2,4,6)


def exact_allocation(errors,choices=(0,1,2)):
    """Minimum sum MSE under an integer block-pair budget, with assignments."""
    n=len(errors);limit=2*n
    dp=np.full(limit+1,np.inf);dp[0]=0
    previous=[]
    for row in errors:
        nxt=np.full(limit+1,np.inf);chosen=np.full(limit+1,-1,dtype=np.int16)
        for k in choices:
            if k:
                cand=dp[:-k]+row[k];current=nxt[k:];better=cand<current
                current[better]=cand[better];chosen[k:][better]=k
            else:
                cand=dp+row[0];better=cand<nxt;nxt[better]=cand[better];chosen[better]=0
        dp=nxt;previous.append(chosen)
    result=[]
    for budget in range(limit+1):
        spent=int(np.argmin(dp[:budget+1]));score=float(dp[spent]);remaining=spent;assignment=[]
        for chosen in reversed(previous):
            k=int(chosen[remaining])
            if k<0:raise AssertionError('Missing dynamic-programming backpointer')
            assignment.append(k);remaining-=k
        assignment.reverse()
        if remaining or sum(assignment)!=spent:raise AssertionError('Invalid allocation budget')
        direct=float(errors[np.arange(n),assignment].sum())
        if not np.isclose(direct,score,rtol=1e-12,atol=1e-15):raise AssertionError('Allocation value mismatch')
        result.append({'budget_units':budget,'spent_units':spent,'sum_mse':score,'assignment':assignment})
    return result


def blind_histogram(errors):
    """Best expected error over uniformly permuted, source-calibrated histograms."""
    n=len(errors);means=errors.mean(axis=0);best=[None]*(2*n+1)
    for n6 in range(n+1):
        for n4 in range(n-n6+1):
            hist=[n-n4-n6,n4,n6];cost=n4+2*n6;score=float(np.dot(hist,means))
            if best[cost] is None or score<best[cost]['sum_mse']:
                best[cost]={'spent_units':cost,'sum_mse':score,'histogram':hist}
    result=[]
    for budget in range(2*n+1):
        candidates=[x for x in best[:budget+1] if x is not None]
        chosen=min(candidates,key=lambda x:(x['sum_mse'],x['spent_units']))
        result.append(dict(chosen,budget_units=budget))
    return result


def checks():
    errors=np.array([[.10,.04,.03],[.02,.025,.01],[.20,.10,.12]])
    for choices in ((0,1,2),(0,2)):
        result=exact_allocation(errors,choices)
        for r in result:
            brute=min(sum(errors[i,k] for i,k in enumerate(a)) for a in itertools.product(choices,repeat=3) if sum(a)<=r['budget_units'])
            assert abs(brute-r['sum_mse'])<1e-12
    oracle=exact_allocation(errors);blind=blind_histogram(errors)
    assert all(a['sum_mse']<=b['sum_mse']+1e-12 for a,b in zip(oracle,blind))
    same=np.tile([.10,.08,.05],(3,1))
    assert all(abs(a['sum_mse']-b['sum_mse'])<1e-12 for a,b in zip(exact_allocation(same),blind_histogram(same)))
    return 'Exact DP agrees with exhaustive3-image allocation; source-informed optimum dominates expected shuffled histograms; identical benefit curves have zero allocation premium.'


def main():
    check=checks()
    source=BASE/'div2k100_epoch020/analysis.json';data=json.loads(source.read_text())
    mp=BASE/'depth_macs/analysis.json';mac=json.loads(mp.read_text())
    costs={r['depth']:r['neural_decoder_macs'] for r in mac['rows']}
    unit=costs[4]-costs[2]
    if costs[6]-costs[4]!=unit:raise ValueError('Depth cost is not equally spaced')
    results=[]
    for rate in (.1,.2,.4):
        record=next(r for r in data['matched_rate'] if (r['rate_field'],r['interpolator'],r['target_bpp'])==('payload_bpp','linear',rate))
        values=record['quality_per_image']
        ids=[i for i,v in values.items() if all(v[str(d)] is not None for d in DEPTHS)]
        if not ids:
            results.append({'rate':rate,'n':0,'images':[],'state':'no_common_support'});continue
        psnr=np.array([[values[i][str(d)] for d in DEPTHS] for i in ids])
        errors=10**(-psnr/10);n=len(ids)
        oracle=exact_allocation(errors);endpoints=exact_allocation(errors,(0,2));blind=blind_histogram(errors)
        curves=[]
        for a,b,c in zip(oracle,endpoints,blind):
            if a['sum_mse']>min(b['sum_mse'],c['sum_mse'])+1e-12:raise AssertionError('Oracle bound violated')
            def summarize(r):
                return dict(r,pooled_psnr_db=float(-10*np.log10(r['sum_mse']/n)),
                    mean_neural_decoder_gmac=(costs[2]+unit*r['spent_units']/n)/1e9,
                    mean_depth=2+2*r['spent_units']/n,
                    saving_vs_d6_percent=100*(1-(costs[2]+unit*r['spent_units']/n)/costs[6]))
            curves.append({'budget_units':a['budget_units'],
                'allowed_mean_depth':2+2*a['budget_units']/n,
                'allowed_mean_neural_decoder_gmac':(costs[2]+unit*a['budget_units']/n)/1e9,
                'all_three':summarize(a),'without_d4':summarize(b),'blind_histogram':summarize(c),
                'd4_option_value_db':float(10*np.log10(b['sum_mse']/a['sum_mse'])),
                'placement_premium_db':float(10*np.log10(c['sum_mse']/a['sum_mse']))})
        results.append({'rate':rate,'n':n,'images':ids,'state':'complete','curves':curves,
            'quality_db':psnr.tolist(),'uniform_pooled_psnr_db':{str(d):float(-10*np.log10(errors[:,k].mean())) for k,d in enumerate(DEPTHS)}})
    result={'scope':'Source-informed whole512-crop allocation using interpolated same-payload-rate quality, epoch20 D2/D4/D6 only. No D12 provenance confound; no trained router, spatial within-crop adaptation, actual mixed codec or latency result.',
        'objective':'Minimum sum per-image RGB MSE; display pooled-crop PSNR, not mean per-image PSNR. Choices share equal target actual payload bpp via per-image log-rate interpolation.',
        'cost':'Architectural neural decoder Conv2d MACs, including entropy recovery networks. Integer units correspond to one additional two-block pair on one crop.',
        'blind_control':'Best histogram chosen from this evaluation cohort using aggregate MSE, then expected MSE under uniform permutation of its labels. This is source-calibrated and optimistic, not a validation-fitted deployable baseline or expected PSNR.',
        'limitation':'No extrapolation; rate cohorts differ. Single training seed and interim checkpoint. Main plotted gains are deterministic conditional diagnostics; no confidence interval or generalization claim.',
        'results':results,'checks':check,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,mp,Path(__file__))}}
    out=BASE/'depth_allocation';out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# D4 gerçekten ayrı bir seçenek olarak değer katıyor mu?','',
        'Bu analiz, aynı payload rate\'e interpolate edilen epoch20 D2/D4/D6 sonuçlarından bir allocation üst sınırı çıkarır. Seçim biriminin512×512 crop olması nedeniyle bunu görüntü-içi patch router sonucu saymıyoruz.',
        'Amaç ortalama MSE; gösterilen pooled-crop PSNR, önceki mean-image PSNR tablolarıyla aynı istatistik değildir.',
        '', '| Payload bpp | n | Ortalama en fazla4 blokta D4 seçenek değeri (dB) | Aynı bütçede placement premium (dB) |', '|---:|---:|---:|---:|']
    for r in results:
        if not r['n']:continue
        x=r['curves'][r['n']]
        lines.append(f"| {r['rate']:.1f} | {r['n']} | {x['d4_option_value_db']:.5f} | {x['placement_premium_db']:.5f} |")
    lines+=['','D4 seçenek değeri: D2/D6 ile ulaşılabilen en iyi sonuç ile D2/D4/D6 optimumu arasındaki fark. Placement premium: aynı evaluation cohort\'unda en iyi histogramın kör permütasyon beklentisi ile kaynak bilgili optimum arasındaki fark.',
        'Her iki kontrol de aynı izin verilen toplam MAC bütçesini kullanır; kullanılmayan bütçe zorla tüketilmez. Kör histogramın kendisi kaynakla kalibre edilir, dolayısıyla deployed kontrol değildir.',
        'Gerçek MLP eğitimi, test ayrımı, patch entropy reset/seam maliyetleri ve latency bu analizde yok. Sonuç, sonraki deneyin kapasite seçeneklerini gerekçelendirmek içindir.']
    (out/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'out':str(out),'cohorts':[(r['rate'],r['n']) for r in results]}))


if __name__=='__main__':main()
