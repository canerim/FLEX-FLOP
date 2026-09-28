"""Build an isolated pinned rANS extension with an owned decoded-array accessor.

No pip install and no changes to upstream or the running training environment.
Only the Python access to decoded symbols is added; coding/CDF kernels stay
byte-for-byte upstream. The build directory records original and patched hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
UPSTREAM=ROOT/'upstream'
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'
DEFAULT_OUT=ROOT/'research/reference_entropy_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(upstream,out):
    head=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError(f'Unexpected upstream commit: {head}')
    if subprocess.check_output(['git','-C',str(upstream),'diff','--','src/cpp']):
        raise ValueError('Pinned upstream entropy sources have local changes')
    source=out/'source';source.mkdir(parents=True,exist_ok=True)
    original={}
    for path in sorted((upstream/'src/cpp/py_rans').iterdir()):
        if path.suffix in ('.cpp','.h'):
            original[path.name]=digest(path);shutil.copy2(path,source/path.name)
    shutil.copy2(upstream/'LICENSE.txt',out/'UPSTREAM_LICENSE.txt')
    shutil.copy2(upstream/'NOTICE.txt',out/'UPSTREAM_NOTICE.txt')
    p=source/'py_rans.h';s=p.read_text();needle='    std::shared_ptr<std::vector<int8_t>> get_decoded_tensor_cpp();'
    if s.count(needle)!=1:raise ValueError('Unexpected decoder declaration')
    p.write_text(s.replace(needle,needle+'\n    py::array_t<int8_t> get_decoded_tensor();'))
    p=source/'py_rans.cpp';s=p.read_text();needle='void RansDecoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,'
    if s.count(needle)!=1:raise ValueError('Unexpected decoder implementation')
    accessor='''py::array_t<int8_t> RansDecoder::get_decoded_tensor()
{
    auto decoded = get_decoded_tensor_cpp(); // waits for every active worker
    py::array_t<int8_t> output({ m_current_decoded_tensor_size }, { sizeof(int8_t) });
    std::copy_n(decoded->data(), m_current_decoded_tensor_size, output.mutable_data());
    return output; // owned copy; later decodes cannot overwrite this array
}

'''
    p.write_text(s.replace(needle,accessor+needle))
    p=source/'bind.cpp';s=p.read_text();needle='        .def("decode_z", &RansDecoder::decode_z)'
    if s.count(needle)!=1:raise ValueError('Unexpected Python binding')
    p.write_text(s.replace(needle,needle+'\n        .def("get_decoded_tensor", &RansDecoder::get_decoded_tensor)'))
    return original,{p.name:digest(p) for p in sorted(source.iterdir())}


def build(upstream,out):
    import pybind11
    out.mkdir(parents=True,exist_ok=True)
    original,patched=prepare(upstream,out)
    target=out/('MLCodec_extensions_cpp'+sysconfig.get_config_var('EXT_SUFFIX'))
    cxx=shutil.which('g++')
    if not cxx:raise RuntimeError('g++ is required')
    cmd=[cxx,'-O3','-shared','-std=c++17','-fPIC','-pthread','-Wall','-Wextra','-Werror',
         '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],
         *[str(p) for p in sorted((out/'source').glob('*.cpp'))],'-o',str(target)]
    subprocess.run(cmd,check=True)
    record=dict(upstream_commit=COMMIT,original_sources_sha256=original,patched_sources_sha256=patched,
                patch='Owned NumPy copy accessor for current decoded symbols; coding and CDF algorithms unchanged',
                module_path=str(target),module_sha256=digest(target),command=cmd,python=sys.version,
                pybind11=pybind11.__version__,compiler=subprocess.check_output([cxx,'--version'],text=True).splitlines()[0],
                build_script_sha256=digest(Path(__file__)))
    (out/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'module':str(target),'sha256':record['module_sha256'],'upstream_unchanged':True}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream',type=Path,default=UPSTREAM)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args();build(args.upstream,args.out)
