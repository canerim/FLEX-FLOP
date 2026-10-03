"""Build the pinned CPU rANS extension with an owned decoded-array accessor.

The patch changes only Python ownership of the decoder's output array. It
does not change the rANS coder or its CDFs. The source is Microsoft's pinned
DCVC-UF commit, fetched separately by the user; no prebuilt ABI is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

COMMIT = 'cbdae87a5445114cdc7f48816da63ea80bdeac40'
ORIGINAL = {
    'bind.cpp': 'e03c3a87e63147ee7db111b5ab2dbefc781d5334954e5a3e8e26c92b60939543',
    'py_rans.cpp': '02d0b46126c98d5a97299ca54f4afa9c3626c0f46308e45760b6e1c213ad506b',
    'py_rans.h': 'ed7fc6dd398de623e48e1907893e8d3a8ae97c98d27adcf7a653c7767c1e983f',
    'rans.cpp': '846d5924f4ca651c10f26d8e49d3349ebdf8d54cb678afc4fd07ed5452634e9e',
    'rans.h': '82e262a5cf28e0ef42e725f87cd689d761f5d6a8bd94aab21724f8f124f4fd65',
}
PATCHES = {
    'py_rans.h': (
        '    std::shared_ptr<std::vector<int8_t>> get_decoded_tensor_cpp();\n',
        '    std::shared_ptr<std::vector<int8_t>> get_decoded_tensor_cpp();\n'
        '    py::array_t<int8_t> get_decoded_tensor();\n'),
    'py_rans.cpp': (
        'void RansDecoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,\n',
        'py::array_t<int8_t> RansDecoder::get_decoded_tensor()\n'
        '{\n'
        '    auto decoded = get_decoded_tensor_cpp(); // waits for every active worker\n'
        '    py::array_t<int8_t> output({ m_current_decoded_tensor_size }, { sizeof(int8_t) });\n'
        '    std::copy_n(decoded->data(), m_current_decoded_tensor_size, output.mutable_data());\n'
        '    return output; // owned copy; later decodes cannot overwrite this array\n'
        '}\n\n'
        'void RansDecoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,\n'),
    'bind.cpp': (
        '        .def("decode_z", &RansDecoder::decode_z)\n',
        '        .def("decode_z", &RansDecoder::decode_z)\n'
        '        .def("get_decoded_tensor", &RansDecoder::get_decoded_tensor)\n'),
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    out = args.out.resolve()
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f'Expected upstream commit {COMMIT}, found {head}')
    import pybind11
    source = upstream/'src/cpp/py_rans'
    for name, expected in ORIGINAL.items():
        if sha(source/name) != expected:
            raise RuntimeError(f'Original Microsoft source hash differs: {name}')
    out.mkdir(parents=True, exist_ok=True)
    copied = out/'source'
    copied.mkdir(exist_ok=True)
    patched = {}
    for name in ORIGINAL:
        target = copied/name
        content = (source/name).read_text()
        if name in PATCHES:
            old, new = PATCHES[name]
            if content.count(old) != 1:
                raise RuntimeError(f'Patch anchor missing or ambiguous: {name}')
            content = content.replace(old, new)
        target.write_text(content)
        patched[name] = sha(target)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    module = out/('MLCodec_extensions_cpp' + suffix)
    command = [shutil.which('g++') or 'g++', '-O3', '-shared', '-std=c++17',
               '-fPIC', '-pthread', '-Wall', '-Wextra', '-Werror',
               '-I'+pybind11.get_include(), '-I'+sysconfig.get_paths()['include'],
               str(copied/'bind.cpp'), str(copied/'py_rans.cpp'),
               str(copied/'rans.cpp'), '-o', str(module)]
    subprocess.run(command, check=True)
    manifest = {
        'upstream_commit': COMMIT, 'original_sources_sha256': ORIGINAL,
        'patched_sources_sha256': patched,
        'patch': 'Owned NumPy copy accessor for current decoded symbols; coding and CDF algorithms unchanged',
        'module_path': module.name, 'module_sha256': sha(module),
        'command': command, 'python': sys.version,
        'pybind11': pybind11.__version__,
        'compiler': subprocess.check_output([command[0], '--version'], text=True).splitlines()[0],
        'build_script_sha256': sha(__file__),
    }
    (out/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'extension': str(module), 'sha256': manifest['module_sha256'],
                      'manifest': str(out/'build_manifest.json')}))


if __name__ == '__main__':
    main()
