"""Check the published paper bundle without GPU or training checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/refresh20260927"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-pdfs",action="store_true",help="Also inspect compiled PDFs with Poppler tools")
    args=ap.parse_args()
    checked=[]
    for name, expected in json.loads((DATA/"bundle_manifest.json").read_text()).items():
        assert digest(DATA/name)==expected,f"Changed bundled source: {name}"
        checked.append(str((DATA/name).relative_to(ROOT)))
    d=json.loads((DATA/"analysis.json").read_text())
    with (DATA/"samples.csv").open() as f:samples=list(csv.DictReader(f))
    for row in d["summary"]:
        group=[r for r in samples if float(r["budget"])==row["budget"] and r["rule"]==row["rule"]]
        assert len(group)==row["n"]
        assert abs(sum(float(r["saving"]) for r in group)/len(group)-row["mean"])<1e-10
        assert sum(float(r["rgb_loss"])>row["budget"]+1e-4 for r in group)==row["rgb_over"]
        assert sum(float(r["yuv_loss"])>row["budget"]+1e-4 for r in group)==row["yuv_over"]
    cap_data=json.loads((DATA/"delivered_frontier_audit.json").read_text())
    for summary in cap_data["summary"]:
        group=[r for r in cap_data["rows"] if r["cap"]==summary["cap"] and r["rule"]==summary["rule"]]
        assert len(group)==summary["n"]==265
        assert len({(r["sequence"],r["qp"]) for r in group})==265
        assert abs(sum(r["saving"] for r in group)/265-summary["mean"])<1e-10
        assert abs(sum(r["loss"] for r in group)/265-summary["mean_loss"])<1e-10
        assert sum(r["fallback"] for r in group)==summary["fallbacks"]
        assert all(r["loss"]<=r["cap"]+cap_data["tolerance_db"] for r in group)
        assert all(r["loss"]==r["saving"]==0 for r in group if r["fallback"])
    influence=json.loads((DATA/"cap_influence_audit.json").read_text())
    assert influence["source_sha256"]==digest(DATA/"delivered_frontier_audit.json")
    assert influence["script_sha256"]==digest(ROOT/"scripts/audit_cap_influence.py")
    for row in influence["rows"]:
        reference_contrast=next(r for r in cap_data["contrasts"]
            if r["cap"]==row["cap_db"] and r["contrast"]=="router_minus_dither")
        assert abs(row["paired_frames"]["mean_points"]-reference_contrast["mean"])<1e-10
        for group, n in [(row["paired_frames"],265),(row["sequence_means"],53)]:
            assert group["n"]==n
            assert sum(group[k] for k in ("router_higher","dither_higher","tied"))==n
        assert len(row["leave_one_sequence_out"])==53
        for left in row["leave_one_sequence_out"]:
            values=[r["saving_difference_points"] for r in row["paired_rows"]
                    if r["sequence"]!=left["removed"]]
            assert len(values)==260
            assert abs(sum(values)/len(values)-left["mean_points"])<1e-10
    reference=json.loads((DATA/"reference_audit.json").read_text())["warmstart_matches_official_release"]
    assert reference["compared"]==398 and not reference["missing"] and not reference["changed"]
    design=json.loads((DATA/"design_audit.json").read_text())
    assert design["router"]["inputs"]=="stem,qp"
    assert "NOT MEASURED" in design["projection"]["label"]
    for row in design["parameters"]["rows"]:
        assert row["decoder_parameters"]==1767168+1038720*row["depth"]
        assert row["total_parameters"]==row["decoder_parameters"]+27947520
    architecture=json.loads((DATA/"depth_architecture_check.json").read_text())
    assert not architecture["gpu_initialized"]
    assert len(architecture["rows"])==6
    for row in architecture["rows"]:
        claimed=next(r for r in design["parameters"]["rows"] if r["depth"]==row["depth"])
        assert all(row[k]==claimed[k] for k in ("decoder_parameters","total_parameters"))
        assert not row["differing_initial_tensors"] and row["all_parameters_trainable"]
    for row in design["projection"]["rows"]:
        f=row["assumed_trunk_time_fraction"];h=row["assumed_extra_overhead_fraction"]
        assert abs(row["relative_time"]-((1-f)+f*row["depth"]/12+h))<1e-12
        assert abs(row["speedup"]*row["relative_time"]-1)<1e-12
    depth=json.loads((DATA/"exit_depth_profile.json").read_text())
    for summary in depth["uniform_summary"]:
        group=[r for r in depth["uniform_rows"] if r["depth"]==summary["depth"] and
               (summary["qp"] is None or r["qp"]==summary["qp"])]
        assert len(group)==summary["n"]
        assert abs(sum(r["padded_rgb_loss_db"] for r in group)/len(group)-summary["mean"])<1e-10
    assert "other_decoders" not in d and "perception" not in d
    metric=json.loads((DATA/'metric_provenance.json').read_text())
    assert digest(ROOT/metric['proof_file'])==metric['proof_sha256']
    proof=json.loads((ROOT/metric['proof_file']).read_text())
    assert proof['strict_checkpoint_load'] and not proof['cuda_initialized']
    assert len(proof['rows'])==2
    for row in proof['rows']:
        value=row['unclipped_ycbcr444']
        assert abs(10*math.log10(value['candidate_mse']/value['reference_mse'])-value['loss_db'])<1e-12
        assert abs(value['loss_db']-row['archived_db_rgb'])<1.1e-6
    figure_count=0
    figure_panels={}
    min_font=float("inf")
    for folder in ["refresh20260927","extended20260927","crossfit20260927","depthmacs20260927"]:
        path=ROOT/"figs"/folder
        for name, expected in json.loads((path/"artifact_manifest.json").read_text()).items():
            assert digest(path/name)==expected,f"Changed figure: {folder}/{name}"
            checked.append(str((path/name).relative_to(ROOT)))
        audit=(json.loads((path/'figure_evidence.json').read_text())['layout_audit'] if folder=='depthmacs20260927'
               else json.loads((path/"layout_audit.json").read_text()))
        for record in audit:
            figure_panels[record['figure']]={t['text'] for t in record['text'] if re.fullmatch('[a-f]',t['text'])}
            assert not record["outside_canvas"],record["figure"]
            expected_width=89 if record['figure'] in {'fig8_active_tiles','figS7_delivered_increment'} else 183
            assert abs(record["width_mm"]-expected_width)<1e-6
            assert record["height_mm"]<=170
            assert "<text" in (path/(record["figure"]+".svg")).read_text()
            min_font=min(min_font,*(t["size_pt"] for t in record["text"]))
            figure_count+=1
    assert figure_count==18
    research=json.loads((ROOT/'data/research20260927/manifest.json').read_text())
    assert digest(ROOT/'scripts/research_figure_paths_20260927.py')==research['helper_sha256']
    research_inputs={}
    for family in research['families']:
        folder=family['folder'];source=ROOT/'data/research20260927'/folder/family['data_file']
        assert digest(source)==family['data_sha256']
        assert digest(ROOT/'scripts'/family['script'])==family['plot_script_sha256']
        for name,expected in family.get('auxiliary_files_sha256',{}).items():
            assert digest(source.parent/name)==expected
            checked.append(str((source.parent/name).relative_to(ROOT)))
        research_inputs[folder]=json.loads(source.read_text());checked.append(str(source.relative_to(ROOT)))
        path=ROOT/'figs/research20260927'/folder
        evidence=json.loads((path/'figure_evidence.json').read_text())
        assert evidence.get('analysis_sha256',evidence.get('association_sha256'))==digest(source)
        assert evidence['script_sha256']==family['plot_script_sha256']
        for name,expected in json.loads((path/'artifact_manifest.json').read_text()).items():
            assert digest(path/name)==expected,f'Changed research figure: {folder}/{name}'
            checked.append(str((path/name).relative_to(ROOT)))
        for record in evidence['layout_audit']:
            figure_panels[record['figure']]={t['text'] for t in record['text'] if re.fullmatch('[a-f]',t['text'])}
            assert not record['outside_canvas'],record['figure']
            assert abs(record['width_mm']-183)<1e-6 and record['height_mm']<=170
            assert '<text' in (path/(record['figure']+'.svg')).read_text()
            min_font=min(min_font,*(t['size_pt'] for t in record['text']))
            figure_count+=1
    assert min_font>=6.5,f'Figure text below the editorial readability floor: {min_font}'
    caption_panels_checked=0
    for tex in (ROOT/'sec').glob('revision*.tex'):
        for block in re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',tex.read_text(),re.S):
            graphic=re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block)
            if not graphic or Path(graphic[1]).suffix!='.pdf':continue
            name=Path(graphic[1]).stem
            assert name in figure_panels,f'Unaudited vector figure in {tex.name}: {name}'
            cited=set()
            for spec in re.findall(r'\\textbf\{([a-f](?:(?:--|,)[a-f])?)\}',block):
                if '--' in spec:
                    first,last=spec.split('--');cited.update(chr(i) for i in range(ord(first),ord(last)+1))
                else:cited.update(spec.split(','))
            assert cited.issubset(figure_panels[name]),f'Caption references a missing panel: {tex.name}/{name}'
            caption_panels_checked+=1
    interim=research_inputs['div2k100_epoch020']
    assert interim['n_cases']==len(interim['rows'])==2000 and interim['n_images']==100
    assert interim['manifest']['epoch_shallow']==20
    assert interim['manifest']['source_analysis_disabled_in_decoder']
    assert len({(r['depth'],r['image'],r['qp']) for r in interim['rows']})==2000
    for row in interim['rows']:
        assert abs(row['container_bpp']-row['payload_bpp']-88*8/512**2)<1e-12
        assert all(math.isfinite(row[k]) for k in ('payload_bpp','estimated_bpp','psnr_rgb'))
    for row in interim['native_qp_means']:
        selected=[r for r in interim['rows'] if (r['depth'],r['qp'])==(row['depth'],row['qp'])]
        assert len(selected)==100
        for field in ('psnr_rgb','payload_bpp'):
            assert abs(sum(r[field] for r in selected)/100-row[field])<1e-10
    associations=research_inputs['source_features']
    assert len(associations['rows'])==18
    assert associations['quality_analysis_sha256']==digest(ROOT/'data/research20260927/div2k100_epoch020/analysis.json')
    for row in associations['rows']:
        assert row['n']==len(row['images'])==len(row['feature_values'])==len(row['depth_gain_db'])
    for cohort in research_inputs['depth_allocation']['results']:
        assert len(cohort['curves'])==2*cohort['n']+1
        assert all(r['d4_option_value_db']>=-1e-9 and r['placement_premium_db']>=-1e-9 for r in cohort['curves'])
    if 'shared_crossfit_qp32' in research_inputs:
        replay=research_inputs['shared_crossfit_qp32'];rows=replay['rows']
        assert len(rows)==318 and len({(r['sequence'],r['criterion'],r['policy']) for r in rows})==318
        assert all(r['qp']==32 and r['budget']==.1 for r in rows)
        for summary in replay['summaries']:
            selected=[r for r in rows if (r['criterion'],r['policy'])==(summary['criterion'],summary['policy'])]
            assert len(selected)==summary['n']==53
            for metric,stats in summary['metrics'].items():
                assert abs(sum(r[metric] for r in selected)/53-stats['mean'])<1e-10
            for metric,count in summary['above_nominal_target'].items():
                assert sum(r[metric]>.1+replay['nominal_exceedance_tolerance_db'] for r in selected)==count
        router_path=ROOT/'data/research20260927/shared_crossfit_qp32/router_logit_audit.json'
        if router_path.exists():
            router=json.loads(router_path.read_text())
            assert len(router['rows'])==53
            for row in router['rows']:
                assert sum(r['macs'] for r in row['router_layers'])==row['router_conv_linear_macs']
                for comparison in row['comparisons']:
                    replay_row=next(r for r in rows if (r['sequence'],r['criterion'],r['policy'])==(row['sequence'],comparison['criterion'],'router'))
                    assert comparison['archived_map']==replay_row['map']
                    assert comparison['changed_tiles']==sum(a!=b for a,b in zip(comparison['fresh_map'],comparison['archived_map']))
        stream_path=ROOT/'data/research20260927/shared_crossfit_qp32/shared_stream_preflight.json'
        if stream_path.exists():
            stream=json.loads(stream_path.read_text())
            assert stream['all_exact'] and not stream['cuda_initialized']
            assert len(stream['cases'])==8 and len(stream['malformed_rejections'])==7
            assert stream['decoder_ready']['identities']['source_analysis_disabled']
            assert stream['identities']['front_tensors_equal']==255
            for row in stream['cases']:
                assert row['reconstruction_exact'] and row['latents_symbols_indexes_exact'] and row['source_analysis_disabled']
                assert row['container_bytes']==sum(row[k] for k in ('map_bytes','outer_header_bytes','inner_header_bytes','payload_bytes'))
                assert row['map_bytes']==(len(row['map'])+3)//4
            for name in {r['sequence'] for r in stream['cases']}:
                group=[r for r in stream['cases'] if r['sequence']==name]
                assert len(group)==4 and len({r['inner_stream_sha256'] for r in group})==1
    cohort_path=ROOT/'data/research20260927/shared_crossfit_qp32/cohort_similarity_audit.json'
    if cohort_path.exists():
        cohort=json.loads(cohort_path.read_text());fp={r['sequence']:r for r in cohort['fingerprints']}
        assert len(fp)==53 and len(cohort['pairs'])==1378
        assert len({(r['first'],r['second']) for r in cohort['pairs']})==1378
        candidates=[]
        for pair in cohort['pairs']:
            a,b=fp[pair['first']],fp[pair['second']]
            assert len(a['phash63_bits'])==len(b['phash63_bits'])==63
            assert pair['phash_hamming']==sum(x!=y for x,y in zip(a['phash63_bits'],b['phash63_bits']))
            assert pair['folds']==[a['fold'],b['fold']]
            expected=pair['phash_hamming']<=6 and pair['luma_correlation'] is not None and pair['luma_correlation']>=.98
            assert pair['similarity_candidate']==expected
            if expected:candidates.append(pair)
        assert candidates==cohort['similarity_candidates'] and len(candidates)==2
        assert all(p['crosses_folds'] and not p['same_raw_frame'] for p in candidates)
        groups_path=cohort_path.parent/'proposed_content_groups.json'
        groups=json.loads(groups_path.read_text())
        assert groups['n_groups']==len(groups['groups'])==51
        assert sorted(n for g in groups['groups'] for n in g)==sorted(fp)
        membership={n:i for i,g in enumerate(groups['groups']) for n in g}
        assert all(membership[p['first']]==membership[p['second']] for p in candidates)
        sensitivity=json.loads((cohort_path.parent/'cluster_sensitivity.json').read_text())
        assert sensitivity['source_sha256']['groups']==digest(groups_path)
        assert sensitivity['source_sha256']['replay']==digest(cohort_path.parent/'analysis.json')
        assert len(sensitivity['rows'])==14
        for row in sensitivity['rows']:
            assert set(row['per_sequence'])==set(fp)
            assert abs(sum(row['per_sequence'].values())/53-row['mean'])<1e-10
            if row['metric'] in ('mean_router_minus_dither_saving_points','q90_router_minus_dither_saving_points'):
                assert row['cluster_ci95'][0]<0<row['cluster_ci95'][1]
    if 'patch_control_epoch020' in research_inputs:
        patch=research_inputs['patch_control_epoch020'];rows=patch['rows']
        assert patch['cases']==240 and len(rows)==960 and len(patch['images'])==16
        assert len({(r['depth'],r['image'],r['qp'],r['variant']) for r in rows})==960
        for row in rows:
            headers=88 if row['variant']=='full' else 352
            assert abs(row['container_bpp']-row['payload_bpp']-headers*8/512**2)<1e-12
            assert abs(row['psnr_rgb']+10*math.log10(row['mse_rgb']))<1e-9
        windows=ROOT/'data/research20260927/patch_control_epoch020'
        assert digest(windows/'qualitative_windows.npz')==json.loads((windows/'qualitative_windows.json').read_text())['asset_sha256']
        checked.extend(str((windows/name).relative_to(ROOT)) for name in ('qualitative_windows.npz','qualitative_windows.json'))
    if 'native_padding' in research_inputs:
        padding=research_inputs['native_padding']
        assert padding['n_cases']==240 and len(padding['rows'])==1200
        for row in padding['matched_rate']:
            values=list(row['per_image_native_minus_pad64_db'].values())
            assert len(values)==row['n']==row['native_minus_pad64_psnr_db']['n']
            if values:assert abs(sum(values)/len(values)-row['native_minus_pad64_psnr_db']['mean'])<1e-10
    if 'region_merge' in research_inputs:
        bank=research_inputs['region_merge'];rows=bank['rows']
        assert bank['n_cases']==80 and bank['n_profiles']==len(rows)==800
        assert len({(r['image'],r['qp'],r['profile']) for r in rows})==800
        for row in rows:
            count=row['n_regions']
            assert row['container_bytes']==row['payload_bytes']+88*count+12+4*count
            assert count==(2 if row['merged'] else 4)
        for row in bank['matched_rate']:
            values=list(row['per_image_gain_db'].values())
            assert len(values)==row['n']==row['phase_averaged_gain_db']['n']
            if values:assert abs(sum(values)/len(values)-row['phase_averaged_gain_db']['mean'])<1e-10
    if 'component_interventions' in research_inputs:
        intervention=research_inputs['component_interventions']
        assert intervention['n_sequences']==len(intervention['cases'])==53
        assert intervention['n_outputs']==sum(len(r['rows']) for r in intervention['cases'])==212
        assert all(r['head_replay_exact']==[True,True] and r['baseline_matches_complete_replay'] for r in intervention['cases'])
        for summary in intervention['summaries']:
            selected=[r for r in intervention['contrasts'] if r['variant']==summary['variant']]
            assert len(selected)==53
            for metric,stats in summary['metrics'].items():assert abs(sum(r[metric] for r in selected)/53-stats['mean'])<1e-10
    if 'released_anchor_qp32' in research_inputs:
        anchors=research_inputs['released_anchor_qp32']
        assert anchors['n_sequences']==len(anchors['cases'])==53
        assert anchors['n_policy_cases']==len(anchors['policy_rows'])==318
        for case in anchors['cases']:
            assert abs(case['released_cropped_rgb_psnr']-case['e15_cropped_rgb_psnr']-case['e15_full_rgb_loss_vs_released_db'])<1e-10
        for summary in anchors['policy_summaries']:
            selected=[r for r in anchors['policy_rows'] if (r['criterion'],r['policy'])==(summary['criterion'],summary['policy'])]
            assert len(selected)==53
            for metric,stats in summary['metrics'].items():assert abs(sum(r[metric] for r in selected)/53-stats['mean'])<1e-10
    calibration_source=json.loads((ROOT/'figs/crossfit20260927/source_manifest.json').read_text())
    assert calibration_source['analysis_sha256']==digest(ROOT/'data/crossfit20260927/analysis.json')
    assert calibration_source['plot_script_sha256']==digest(ROOT/'scripts/plot_crossfit_control_20260927.py')
    mac_source=json.loads((ROOT/'figs/depthmacs20260927/figure_evidence.json').read_text())
    assert mac_source['analysis_sha256']==digest(ROOT/'data/depthmacs20260927/analysis.json')
    assert mac_source['script_sha256']==digest(ROOT/'scripts/plot_depth_macs_20260927.py')
    mac=json.loads((ROOT/'data/depthmacs20260927/analysis.json').read_text())
    assert [r['depth'] for r in mac['rows']]==[2,4,6,8,10,12]
    assert not mac['cuda_initialized']
    for row in mac['rows']:
        assert sum(r['macs'] for r in row['layers'])==row['encoder_with_reconstruction_macs']
        assert row['neural_decoder_macs']==row['entropy_neural_macs']+row['synthesis_macs']
        assert row['spatial_prior_calls']==3
        assert row['entropy_neural_macs']==mac['rows'][0]['entropy_neural_macs']
    for family in ('adaptive20260927','system20260928'):
        illustration=json.loads((ROOT/'figs'/family/'provenance.json').read_text())
        asset=ROOT/illustration['asset']
        assert digest(asset)==illustration['sha256'],'Changed conceptual illustration'
        checked.append(str(asset.relative_to(ROOT)))
    assert digest(ROOT/'figs/adaptive20260927/adaptive_overview.png')=='45f15d5951b72c1ac8da6c34fd6593d271615a1922b7c3ebfc22b331fec09cf8','Protected Figure 1 changed'
    assert digest(ROOT/'figs/system20260928/shared_exit_mechanism.png')=='abc06d7d677d82934f3aba8c0e50effb0300ab68c4cfbc898cf8cd28caea739a','Protected Figure 2 changed'
    reports={}
    if args.check_pdfs:
        for name in ["main","supplement"]:
            content=subprocess.check_output(["pdftotext",str(ROOT/(name+".pdf")),"-"],text=True)
            pages=[page for page in content.split("\f") if page.strip()]
            assert all(len(page.split())>30 for page in pages),f"Near-blank PDF page in {name}"
            assert "??" not in content,f"Unresolved reference in {name}"
            assert not any(s in content for s in ["NoGAN", "MS-ILLM"]),f"Out-of-scope decoder in {name}"
            log=(ROOT/(name+".log")).read_text()
            assert not re.search(r"undefined (?:references|citations)|Citation .*undefined|^!|Overfull",log,re.M)
            fonts=subprocess.check_output(["pdffonts",str(ROOT/(name+".pdf"))],text=True).splitlines()
            pos=fonts[0].index("emb")
            assert all(line[pos:pos+3]=="yes" for line in fonts[2:] if line.strip()),f"Unembedded font in {name}"
            assert not any("Type 3" in line for line in fonts[2:]),f"Type 3 font in {name}"
            reference_page=next((i+1 for i,page in enumerate(pages) if "\nReferences\n" in page),None)
            assert reference_page is not None,f"Missing bibliography in {name}"
            if name=="main":
                assert reference_page<=9,f"Main text exceeds eight pages: bibliography starts on {reference_page}"
            reports[name]=dict(pages=len(pages),all_fonts_embedded=True,unresolved_references=0,near_blank_pages=0,
                               reference_start_page=reference_page,
                               sha256=digest(ROOT/(name+".pdf")))
    result=dict(bundled_files_checked=len(checked),caption_panels_checked=caption_panels_checked,summary_rows_checked=len(d["summary"]),
                paired_samples=len(samples),delivered_cap_summary_rows_checked=len(cap_data["summary"]),
                delivered_cap_influence_contrasts_checked=len(influence["rows"]),
                released_warmstart_tensors_matched=reference["compared"],
                uniform_depth_summary_rows_checked=len(depth["uniform_summary"]),
                analytical_scenarios_checked=len(design["projection"]["rows"]),
                independently_instantiated_depths_checked=len(architecture["rows"]),
                vector_figure_sets=figure_count,conceptual_ai_illustrations=2,min_figure_font_pt=min_font,
                text_outside_canvas=0,compiled_documents=reports,
                scope="Integrity and internal consistency of the supplied publication bundle, not historical-run reproduction or a new codec evaluation")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
