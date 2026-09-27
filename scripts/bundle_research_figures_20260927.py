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
'native_padding':('plot_native_padding_20260927.py','analysis.json')}
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
        target=dest/folder;target.mkdir(exist_ok=True);shutil.copy2(source,target/filename)
        report=SOURCE/folder/'REPORT_TR.md'
        if report.exists():shutil.copy2(report,target/report.name)
        shutil.copy2(ROOT/'scripts'/script,PAPER/'scripts'/script)
        families.append({'folder':folder,'script':script,'data_file':filename,'data_sha256':sha(target/filename),'plot_script_sha256':sha(PAPER/'scripts'/script)})
    helper='research_figure_paths_20260927.py';shutil.copy2(ROOT/'scripts'/helper,PAPER/'scripts'/helper)
    record={'scope':'Completed CPU research and architecture diagnostics. Intermediate epoch20 is not a final converged depth-bank or runtime result. Source paths inside provenance are historical metadata; rendering requires only bundled data.', 'families':families,'helper_sha256':sha(PAPER/'scripts'/helper)}
    (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'families':[x['folder'] for x in families]}))
if __name__=='__main__':main()
