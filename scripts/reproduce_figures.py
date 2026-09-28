"""Rebuild all measured/schematic vector figures in a clean temporary bundle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('refresh20260927','extended20260927','crossfit20260927','depthmacs20260927','editorial20260928')
SCRIPTS=('build_evidence_tables.py','build_figures.py','build_extended_figures.py','plot_crossfit_control_20260927.py','plot_depth_macs_20260927.py','build_research_figures.py','build_editorial_figures.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    compared=[];tables=[]
    with tempfile.TemporaryDirectory(prefix='flex-vector-reproduction-') as tmp:
        dest=Path(tmp)
        shutil.copytree(ROOT/'scripts',dest/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'data',dest/'data')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for script in SCRIPTS:
            subprocess.run([sys.executable,str(dest/'scripts'/script)],cwd=dest,env=env,check=True,
                stdout=subprocess.DEVNULL)
        evidence=json.loads((ROOT/'data/refresh20260927/evidence_tables_manifest.json').read_text())
        reproduced=json.loads((dest/'data/refresh20260927/evidence_tables_manifest.json').read_text())
        if evidence!=reproduced:raise AssertionError('Non-reproducible evidence table manifest')
        for name,digest in evidence['artifacts'].items():
            relative=Path('data/refresh20260927')/name
            if sha(ROOT/relative)!=digest or sha(dest/relative)!=digest:
                raise AssertionError('Evidence table mismatch: '+name)
            tables.append({'file':str(relative),'sha256':digest,'byte_identical':True})
        research=json.loads((ROOT/'data/research20260927/manifest.json').read_text())
        folders=FOLDERS+tuple('research20260927/'+f['folder'] for f in research['families'])
        for folder in folders:
            expected=json.loads((ROOT/'figs'/folder/'artifact_manifest.json').read_text())
            actual=json.loads((dest/'figs'/folder/'artifact_manifest.json').read_text())
            if expected!=actual:raise AssertionError('Non-reproducible figure manifest: '+folder)
            for name,digest in expected.items():
                a=ROOT/'figs'/folder/name;b=dest/'figs'/folder/name
                if sha(a)!=digest or sha(b)!=digest:raise AssertionError('Artifact mismatch: '+str(a))
                compared.append({'file':str(a.relative_to(ROOT)),'sha256':digest,'byte_identical':True})
    record={'scope':'Current vector figure reproduction using only bundled data/scripts; no codec inference, training weights, source dataset or GPU.',
        'artifact_count':len(compared),'all_byte_identical':True,'artifacts':compared,
        'evidence_text_artifact_count':len(tables),'evidence_text_artifacts':tables,
        'scripts_sha256':{str((ROOT/'scripts'/name).relative_to(ROOT)):sha(ROOT/'scripts'/name) for name in SCRIPTS+('editorial_visuals_20260928.py',)},
        'reproduction_script_sha256':sha(Path(__file__))}
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
