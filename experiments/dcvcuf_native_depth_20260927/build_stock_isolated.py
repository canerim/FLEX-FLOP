"""Build the unmodified native D12 control with the same isolated toolchain."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from build_isolated import ROOT,HERE,UPSTREAM,CUTLASS,COMMIT,CUTLASS_COMMIT,PREFIX,SETUP,sha,utc,output,atomic

OUT=ROOT/'research/native_stock_build_sm86'


def main():
    previous=json.loads((HERE/'compile_report.json').read_text())
    if previous['state']!='compiled':raise ValueError('Patched control must compile first')
    for path,revision in ((UPSTREAM,COMMIT),(CUTLASS,CUTLASS_COMMIT)):
        if output(['git','-C',str(path),'rev-parse','HEAD'])!=revision:raise ValueError('Dependency revision differs')
        if output(['git','-C',str(path),'status','--porcelain']):raise ValueError('Dependency checkout changed')
    if OUT.exists():raise ValueError('Inspect existing stock build before retry')
    OUT.mkdir(parents=True);source=OUT/'source';source.mkdir()
    archive=subprocess.check_output(['git','-C',str(UPSTREAM),'archive',COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(source,filter='data')
    setup=source/PREFIX/'setup_isolated.py';setup.write_text(SETUP)
    if sha(setup)!=previous['setup_sha256']:raise AssertionError('Build flags differ from patched extension')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-12.1',TORCH_CUDA_ARCH_LIST='8.6',
             MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',FLEX_NATIVE_CUTLASS=str(CUTLASS))
    cmd=[str(ROOT/'venv/bin/python'),str(setup),'build_ext','--build-lib',str(OUT/'lib'),'--build-temp',str(OUT/'objects')]
    record={'state':'compiling','started_utc':utc(),'pid':os.getpid(),'upstream_commit':COMMIT,
        'cutlass_commit':CUTLASS_COMMIT,'patch_applied':False,'setup_sha256':sha(setup),'script_sha256':sha(__file__),
        'command':cmd,'target_sm':86,'cuda_visible_devices':'','max_jobs':1,'nvcc_split_compile':1,
        'nvcc_version':output(['/usr/local/cuda-12.1/bin/nvcc','--version']),
        'toolchain_note':previous['toolchain_note'],
        'scope':'Unmodified D12 compilation control only; no CUDA execution, numerical parity or timing claim.'}
    atomic(OUT/'status.json',record)
    with (OUT/'build.log').open('w') as log:
        result=subprocess.run(cmd,cwd=source/PREFIX,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(state='compiled' if result.returncode==0 else 'failed',returncode=result.returncode,finished_utc=utc(),
        artifacts={str(p):sha(p) for p in (OUT/'lib').glob('*.so')},log_sha256=sha(OUT/'build.log'))
    if output(['git','-C',str(UPSTREAM),'status','--porcelain']):raise AssertionError('Training upstream changed')
    record['upstream_unmodified']=True
    atomic(OUT/'status.json',record);atomic(HERE/'stock_compile_report.json',record)
    print(json.dumps(record,indent=2));return result.returncode


if __name__=='__main__':sys.exit(main())
