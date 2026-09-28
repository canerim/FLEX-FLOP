"""Package completed research figure inputs without server paths at render time."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'cvpr2027';SOURCE=ROOT/'docs/research/2026-09-27-six-hour'
FAMILIES={'div2k100_epoch020':('plot_reference_validation_20260927.py','analysis.json'),
'source_features':('plot_source_associations_20260927.py','association.json'),
'depth_allocation':('plot_depth_allocation_20260927.py','analysis.json'),
'patch_geometry':('plot_patch_geometry_20260927.py','analysis.json'),
'shared_crossfit_qp32':('plot_crossfit_replay_20260927.py','analysis.json'),
'native_padding':('plot_native_padding_20260927.py','analysis.json'),
'patch_control_epoch020':('plot_patch_control_20260927.py','analysis.json'),
'region_merge':('plot_region_merge_20260927.py','analysis.json'),
'native_execution':('plot_native_execution_20260928.py','analysis.json'),
'component_interventions':('plot_component_interventions_20260928.py','analysis.json'),
'released_anchor_qp32':('plot_released_anchor_20260928.py','analysis.json')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=PAPER/'data/research20260927';dest.mkdir(parents=True,exist_ok=True)
    families=[]
    for folder,(script,filename) in FAMILIES.items():
        source=SOURCE/folder/filename
        if not source.exists():continue
        # Only package a result after its original figure was generated and inspected.
        proof=SOURCE/folder/'figure_evidence.json'
        if not proof.exists():continue
        data=json.loads(source.read_text())
        if folder=='div2k100_epoch020':assert data['n_cases']==2000
        if folder=='shared_crossfit_qp32':assert len(data['rows'])==318
        if folder=='native_padding':assert data['n_cases']==240
        if folder=='patch_control_epoch020':assert data['cases']==240
        if folder=='region_merge':assert data['n_cases']==80 and data['n_profiles']==800
        if folder=='component_interventions':assert data['n_sequences']==53 and data['n_outputs']==212
        if folder=='released_anchor_qp32':assert data['n_sequences']==53 and data['n_policy_cases']==318
        target=dest/folder;target.mkdir(exist_ok=True);shutil.copy2(source,target/filename)
        if folder=='patch_control_epoch020':
            for name in ('qualitative_windows.npz','qualitative_windows.json'):
                shutil.copy2(SOURCE/folder/name,target/name)
        report=SOURCE/folder/'REPORT_TR.md'
        if report.exists():shutil.copy2(report,target/report.name)
        shutil.copy2(ROOT/'scripts'/script,PAPER/'scripts'/script)
        entry={'folder':folder,'script':script,'data_file':filename,'data_sha256':sha(target/filename),'plot_script_sha256':sha(PAPER/'scripts'/script)}
        if folder=='shared_crossfit_qp32' and (SOURCE/'router_logit_replay/analysis.json').exists():
            extra=target/'router_logit_audit.json';shutil.copy2(SOURCE/'router_logit_replay/analysis.json',extra)
            entry['auxiliary_files_sha256']={extra.name:sha(extra)}
        if folder=='shared_crossfit_qp32' and (SOURCE/'shared_stream_preflight/analysis.json').exists():
            extra=target/'shared_stream_preflight.json';shutil.copy2(SOURCE/'shared_stream_preflight/analysis.json',extra)
            proof=json.loads(extra.read_text());assert proof['all_exact'] and len(proof['cases'])==8
            entry.setdefault('auxiliary_files_sha256',{})[extra.name]=sha(extra)
        families.append(entry)
    helper='research_figure_paths_20260927.py';shutil.copy2(ROOT/'scripts'/helper,PAPER/'scripts'/helper)
    record={'scope':'Completed CPU research and architecture diagnostics. Intermediate epoch20 is not a final converged depth-bank or runtime result. Source paths inside provenance are historical metadata; rendering requires only bundled data.', 'families':families,'helper_sha256':sha(PAPER/'scripts'/helper)}
    (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'families':[x['folder'] for x in families]}))
if __name__=='__main__':main()
