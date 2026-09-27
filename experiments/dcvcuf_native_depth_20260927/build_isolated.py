"""Compile an isolated SM86 native-depth extension without using a GPU.

Does not install anything or change the training environment/upstream checkout.
Compilation success is not numerical, stream-format, or runtime validation.
"""
from __future__ import annotations
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
HERE=Path(__file__).resolve().parent
UPSTREAM=ROOT/'upstream'
OUT=ROOT/'research/native_depth_build_sm86'
CUTLASS=ROOT/'research/native_dependencies/cutlass_v4.4.1'
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'
CUTLASS_COMMIT='4370102f9dacab813282e1d67722fceb0b90a019'
PREFIX=Path('src/layers/extensions/inference')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def output(cmd):return subprocess.check_output(cmd,text=True).strip()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


SETUP='''# Isolated adaptation of Microsoft setup.py; no GPU query or installation.
import glob
import os
from pathlib import Path
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert not torch.cuda.is_initialized()
base=Path(__file__).resolve().parent
cutlass=Path(os.environ['FLEX_NATIVE_CUTLASS'])
rans=base/'../../../cpp/py_rans'
setup(name='inference_extensions_cuda',ext_modules=[CUDAExtension(
    name='inference_extensions_cuda',
    include_dirs=[str(cutlass/'include'),str(cutlass/'tools/util/include'),str(rans.resolve())],
    sources=sorted(glob.glob('**/*.cpp',recursive=True)+glob.glob('**/*.cu',recursive=True))+
        [str(rans/'rans.cpp'),str(rans/'py_rans.cpp')],
    extra_compile_args={'cxx':['-O3','-Wno-deprecated-declarations'],
        'nvcc':['-DCURRENT_DEVICE_SM=86','-O3','--use_fast_math','--extra-device-vectorization',
                '-gencode=arch=compute_86,code=sm_86','-Wno-deprecated-declarations','--split-compile=1']})],
    cmdclass={'build_ext':BuildExtension})
assert not torch.cuda.is_initialized()
'''


def main():
    if output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'])!=COMMIT:raise ValueError('Upstream revision mismatch')
    if output(['git','-C',str(CUTLASS),'rev-parse','HEAD'])!=CUTLASS_COMMIT:raise ValueError('CUTLASS revision mismatch')
    if output(['git','-C',str(CUTLASS),'status','--porcelain']):raise ValueError('CUTLASS checkout modified')
    if OUT.exists():raise ValueError('Output already exists; inspect it before any retry')
    OUT.mkdir(parents=True)
    source=OUT/'source';source.mkdir()
    archive=subprocess.check_output(['git','-C',str(UPSTREAM),'archive',COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(source,filter='data')
    patch=HERE/'native_depth_draft.patch'
    subprocess.run(['git','apply','--check',str(patch)],cwd=source,check=True)
    subprocess.run(['git','apply',str(patch)],cwd=source,check=True)
    setup=source/PREFIX/'setup_isolated.py';setup.write_text(SETUP)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-12.1',
        TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        FLEX_NATIVE_CUTLASS=str(CUTLASS))
    cmd=[str(ROOT/'venv/bin/python'),str(setup),'build_ext','--build-lib',str(OUT/'lib'),
         '--build-temp',str(OUT/'objects')]
    record={'state':'compiling','started_utc':utc(),'pid':os.getpid(),'upstream_commit':COMMIT,
        'cutlass_tag':'v4.4.1','cutlass_commit':CUTLASS_COMMIT,'patch_sha256':sha(patch),
        'setup_sha256':sha(setup),'command':cmd,'cuda_visible_devices':'','max_jobs':1,
        'target_sm':86,'nvcc_split_compile':1,
        'nvcc_version':output(['/usr/local/cuda-12.1/bin/nvcc','--version']),
        'toolchain_note':'Local CUDA12.1 compiler against PyTorch2.9.1+cu126; minor-version mismatch recorded, not recipe-equivalent runtime evidence.',
        'scope':'Compilation only. GPU correctness, released parity, and wall-time measurements remain pending.'}
    atomic(OUT/'status.json',record)
    with (OUT/'build.log').open('w') as log:
        result=subprocess.run(cmd,cwd=source/PREFIX,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(state='compiled' if result.returncode==0 else 'failed',returncode=result.returncode,finished_utc=utc(),
        artifacts={str(p):sha(p) for p in (OUT/'lib').glob('*.so')},log_sha256=sha(OUT/'build.log'))
    if output(['git','-C',str(UPSTREAM),'status','--porcelain']):raise AssertionError('Upstream is unexpectedly dirty')
    record['upstream_unmodified']=True
    atomic(OUT/'status.json',record)
    atomic(HERE/'compile_report.json',record)
    print(json.dumps(record,indent=2))
    return result.returncode


if __name__=='__main__':sys.exit(main())
