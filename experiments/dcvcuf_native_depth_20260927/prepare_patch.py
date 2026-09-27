"""Prepare a reviewable native-depth patch without changing upstream or building CUDA."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
UPSTREAM=ROOT/'upstream'
HERE=Path(__file__).resolve().parent
OUT=ROOT/'research/native_depth_draft'
PREFIX=Path('src/layers/extensions/inference')
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Pinned source pattern changed: '+old[:100])
    return text.replace(old,new,1)


def main():
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise ValueError('Wrong upstream revision')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--',str(PREFIX)]):
        raise ValueError('Upstream inference sources modified')
    sources={name:(UPSTREAM/PREFIX/name).read_text() for name in ('dmci_proxy.h','dmci_proxy.cpp')}
    h=sources['dmci_proxy.h'];cpp=sources['dmci_proxy.cpp']
    h=replace_once(h,'#include "layers_proxy.h"','#include "layers_proxy.h"\n#include <vector>')
    members=''.join(f'    DepthConvBlockProxy m_conv{i};\n' for i in range(1,13))
    h=replace_once(h,members,'    std::vector<DepthConvBlockProxy> m_blocks;\n')
    h=replace_once(h,'    at::Tensor m_q_scale_enc[g_qp_num];','    bool m_parameters_set{ false };\n    at::Tensor m_q_scale_enc[g_qp_num];')
    cpp=replace_once(cpp,'#include "dmci_proxy.h"','#include "dmci_proxy.h"\n#include "depth_keys.h"')
    for method in ('forward','pre_allocate_tensors'):
        old=''.join(f'    out = m_conv{i}.{method}(out);\n' for i in range(1,13))
        new=f'    for (auto& block : m_blocks) {{\n        out = block.{method}(out);\n    }}\n'
        cpp=replace_once(cpp,old,new)
    cpp=replace_once(cpp,''.join(f'    m_conv{i}.release_tensors();\n' for i in range(1,13)),
                     '    for (auto& block : m_blocks) {\n        block.release_tensors();\n    }\n')
    old=''.join(f'    m_conv{i}.set_param(get_submodule_state_dict(state_dict, "dec_1.{i}."));\n' for i in range(1,13))
    cpp=replace_once(cpp,old,'    m_blocks.resize(depth);\n    for (int i = 0; i < depth; ++i) {\n        m_blocks[i].set_param(get_submodule_state_dict(\n            state_dict, "dec_1." + std::to_string(i + 1) + "."));\n    }\n')
    cpp=replace_once(cpp,'    m_up.set_param(get_submodule_state_dict(state_dict, "dec_1.0."), true);',
                     '    const int depth = flex_depth::infer_synthesis_depth(state_dict);\n    m_up.set_param(get_submodule_state_dict(state_dict, "dec_1.0."), true);')
    cpp=replace_once(cpp,'    m_device = state_dict.at("q_scale_dec").device();',
                     '    TORCH_CHECK(!m_parameters_set,\n                "Use a fresh DMCIProxy for each checkpoint/depth; "\n                "captured graphs cannot be reused with new weights");\n    m_device = state_dict.at("q_scale_dec").device();')
    for expression in ('x.size(2), x.size(3)','height, width'):
        cpp=replace_once(cpp,f'    pre_allocate_tensors({expression});',
                         f'    TORCH_CHECK(m_parameters_set, "Initialize this proxy with set_param before coding");\n    pre_allocate_tensors({expression});')
    cpp=replace_once(cpp,'    pre_allocate_tensors(x.size(2), x.size(3));',
                     '    TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && x.size(1) == 3,\n                "Native depth draft supports one three-channel image per call; "\n                "expert batching requires a separate implementation");\n    pre_allocate_tensors(x.size(2), x.size(3));')
    tail='        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.cdf_length")), 1);\n}\n\nvoid DMCIProxy::clear_cuda_graph()'
    cpp=replace_once(cpp,tail,'        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.cdf_length")), 1);\n    m_parameters_set = true;\n}\n\nvoid DMCIProxy::clear_cuda_graph()')
    folder=OUT/PREFIX;folder.mkdir(parents=True,exist_ok=True)
    edited={'dmci_proxy.h':h,'dmci_proxy.cpp':cpp,'depth_keys.h':(HERE/'depth_keys.h').read_text()}
    patch=''
    for name,text in edited.items():
        (folder/name).write_text(text)
        patch+=''.join(difflib.unified_diff(sources.get(name,'').splitlines(True),text.splitlines(True),
                                          fromfile='a/'+str(PREFIX/name) if name in sources else '/dev/null',
                                          tofile='b/'+str(PREFIX/name)))
    (HERE/'native_depth_draft.patch').write_text(patch)
    for name in ('LICENSE.txt','NOTICE.txt'):shutil.copyfile(UPSTREAM/name,OUT/name)
    executable=OUT/'check_depth_keys'
    cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(HERE),str(HERE/'check_depth_keys.cpp'),'-o',str(executable)]
    subprocess.run(cmd,check=True)
    check=json.loads(subprocess.check_output([str(executable)],text=True))
    # Verify that the prepared patch can be applied to exactly the clean source.
    subprocess.run(['git','-C',str(UPSTREAM),'apply','--check',str(HERE/'native_depth_draft.patch')],check=True)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--',str(PREFIX)]):raise AssertionError('Upstream unexpectedly changed')
    record={'status':'source_patch_prepared; full CUDA extension NOT compiled or executed',
            'upstream_commit':COMMIT,'original_sha256':{name:sha(UPSTREAM/PREFIX/name) for name in sources},
            'patched_sha256':{name:sha(folder/name) for name in edited},'patch_sha256':sha(HERE/'native_depth_draft.patch'),
            'helper_check':check,'compiler_command':cmd,'upstream_unmodified':True,
            'gpu_validation_required':['D12 stock-vs-patched native parity','D2/4/6/8/10/12 actual retained-block execution',
                                       'Fresh decoder process with no compress initialization','Shape/QP changes and graph recapture',
                                       'Serial expert-bank switching and output-lifetime checks','Actual bytes and paired end-to-end wall time']}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    (HERE/'preparation_report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
