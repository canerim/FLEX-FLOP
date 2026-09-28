"""Record source-level native encoder dependencies; never derive latency from them."""
import hashlib,json,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
UPSTREAM=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/upstream')
OUT=REPO/'docs/research/2026-09-27-six-hour/native_execution'
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'


def main():
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('Upstream revision differs')
    names=['src/layers/extensions/inference/dmci_proxy.cpp','src/layers/extensions/inference/memory_pool.h','src/models/image_model.py']
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--',*names]):raise ValueError('Audited native source changed')
    text=(UPSTREAM/names[0]).read_text()
    tokens=['run(lambda_enc_0, m_gexec_enc_0, m_qp);','cudaEventRecord(m_event_y, stream)','m_pending_work = DMCWorkType::Encode;',
        'run(lambda_enc_1, m_gexec_enc_1, m_qp);','m_cv_result.wait(lk,','m_entropy_encoder.get_encoded_stream()',
        'cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), m_event_y)','cudaMemcpyAsync(m_y_to_encode[i]->data()',
        'cudaStreamSynchronize(at::cuda::getCurrentCUDAStream())','m_entropy_encoder.encode_y(','m_entropy_encoder.encode_z(','m_entropy_encoder.flush();']
    anchors={token:text[:text.index(token)].count('\n')+1 for token in tokens}
    result={'scope':'Static dependency and resource audit of pinned native source; no CUDA execution, latency or overlap measurement.',
        'upstream_commit':COMMIT,'source_sha256':{name:hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() for name in names},
        'source_anchors':anchors,
        'encoder_dependencies':['source pad/unshuffle','analysis/hyperanalysis + prior + quantized latent','event_y + notify worker',
            'main-stream synthesis overlaps potential worker-stream index gather/D2H and CPU rANS','wait for entropy worker result','return payload and GPU reconstruction tensor','caller CUDA synchronization for complete wall time'],
        'worker_dependencies':['wait event_y','GPU conditional symbol gather, reverse stages3,2,1,0','device-to-host copies','worker stream synchronization','CPU encode_y, encode_z,flush','signal m_result_ready'],
        'timing_rule':'Charge synchronized elapsed time around the complete API workload. Do not sum separately measured stage times; include output transfer only if the selected deployment contract requires it.',
        'model_bank_rule':'Count source feature extraction, predictor, grouping, model selection, entropy work, region assembly, headers and memory. Mode-map selection does not make independently learned latent representations interchangeable.',
        'pool_observation':'TensorPool is a global object. Acquire/release is mutex-protected, but buffer reuse and cached CUDA graphs require explicit multi-proxy/output-lifetime correctness tests. Static inspection alone does not prove a race or invalid output.',
        'remaining_checks':['stock versus patchedD12 numeric/payload parity','fresh decoder-only initialization','D2/D4/D6 shape/QP recapture','resident expert switching and output lifetime','serial and grouped scheduling latency under the same metric contract'],
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'source_anchors':anchors,'out':str(OUT)}))


if __name__=='__main__':main()
