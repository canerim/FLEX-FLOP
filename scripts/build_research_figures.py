"""Rebuild complete research figure families from the standalone paper bundle."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'data/research20260927/manifest.json').read_text())
    assert sha(ROOT/'scripts/research_figure_paths_20260927.py')==manifest['helper_sha256']
    for family in manifest['families']:
        folder=family['folder'];source=ROOT/'data/research20260927'/folder/family['data_file'];script=ROOT/'scripts'/family['script']
        assert sha(source)==family['data_sha256'];assert sha(script)==family['plot_script_sha256']
        for name,digest in family.get('auxiliary_files_sha256',{}).items():
            assert sha(source.parent/name)==digest
        subprocess.run([sys.executable,str(script)],cwd=ROOT,check=True)
        output=ROOT/'figs/research20260927'/folder
        artifacts={p.name:sha(p) for p in sorted(output.iterdir()) if p.suffix in ('.pdf','.svg','.png')}
        if not artifacts:raise ValueError('No artifacts rendered: '+folder)
        (output/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2)+'\n')
if __name__=='__main__':main()
