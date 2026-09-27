"""Trace architectural Conv2d MACs on meta tensors; no speed claims or GPU use."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
UPSTREAM=ROOT/'upstream'
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'
OUT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour/depth_macs'
DEPTHS=(2,4,6,8,10,12)


def trace(depth,h,w,device='meta'):
    from src.models.image_model import DMCI
    rows=[]
    with torch.device(device):
        net=DMCI().eval()
        net.dec.dec_1=torch.nn.Sequential(*list(net.dec.dec_1.children())[:depth+1])
        handles=[]
        for name,module in net.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                def count(m,inputs,output,name=name):
                    rows.append({'name':name,'input_shape':list(inputs[0].shape),'output_shape':list(output.shape),
                        'macs':output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]})
                handles.append(module.register_forward_hook(count))
        x=torch.zeros(1,3,h,w);qp=torch.tensor([32],dtype=torch.int32)
        with torch.inference_mode():rec=net.forward_one_frame(x,qp,recon_only=True)
        assert rec.shape==x.shape
        for handle in handles:handle.remove()
    groups={}
    for row in rows:
        group=row['name'].split('.')[0];groups[group]=groups.get(group,0)+row['macs']
    assert set(groups)=={'enc','dec','hyper_enc','hyper_dec','y_prior_fusion','y_spatial_prior_reduction',
                         'y_spatial_prior_adaptor_1','y_spatial_prior_adaptor_2','y_spatial_prior_adaptor_3','y_spatial_prior'}
    # The same spatial-prior convolution module is used three times, not once.
    prior_calls=sum(r['name']=='y_spatial_prior.conv.3' for r in rows)
    if prior_calls!=3:raise AssertionError('Spatial-prior reuse not counted correctly')
    prior=sum(v for k,v in groups.items() if k not in ('enc','dec','hyper_enc'))
    return {'depth':depth,'height':h,'width':w,'device':device,'groups':groups,'layers':rows,
        'synthesis_macs':groups['dec'],'analysis_macs':groups['enc']+groups['hyper_enc'],
        'entropy_neural_macs':prior,'neural_decoder_macs':prior+groups['dec'],
        'encoder_with_reconstruction_macs':sum(groups.values()),'spatial_prior_calls':prior_calls}


def main():
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise ValueError('Upstream revision mismatch')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--','src/models','src/layers/layers.py']):
        raise ValueError('Upstream model sources modified')
    sys.path.insert(0,str(UPSTREAM));torch.set_num_threads(1)
    rows=[trace(d,512,512) for d in DEPTHS]
    small_meta=trace(2,64,64);small_cpu=trace(2,64,64,'cpu')
    if small_meta['layers']!=small_cpu['layers']:raise AssertionError('Meta and real CPU Conv2d traces differ')
    for row in rows:
        base=rows[-1]
        for key in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):
            row[key+'_saving_vs_d12_percent']=100*(1-row[key]/base[key])
        if row['analysis_macs']!=base['analysis_macs'] or row['entropy_neural_macs']!=base['entropy_neural_macs']:
            raise AssertionError('Non-synthesis arithmetic unexpectedly changes')
    increments=[rows[i+1]['synthesis_macs']-rows[i]['synthesis_macs'] for i in range(5)]
    if len(set(increments))!=1:raise AssertionError('Constant-width block pair is not constant cost')
    if rows[0]['synthesis_macs']!=small_meta['synthesis_macs']*64:raise AssertionError('Spatial scaling failed')
    assert not torch.cuda.is_initialized()
    files=[Path(__file__),UPSTREAM/'src/models/image_model.py',UPSTREAM/'src/models/common_model.py',UPSTREAM/'src/layers/layers.py']
    result={'scope':'Architectural Conv2d multiply-accumulate accounting; one multiply+accumulate is one MAC. Not CUDA-kernel work, latency, energy, or trained-model quality.',
        'exclusions':['bias addition','nonlinearities','elementwise arithmetic','quantization','CDF/index computation','rANS coding','memory movement','padding/assembly','routing/signalling'],
        'rows':rows,'block_pair_synthesis_macs_512':increments[0],
        'checks':['D2 64x64 meta trace equals real CPU Conv2d trace','Each spatial prior convolution counted three times',
                  'Synthesis cost scales64x from64² to512²','Depth increments constant; analysis and neural entropy cost fixed'],
        'torch':torch.__version__,'cuda_initialized':False,'upstream_commit':COMMIT,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Derinlik hesabında payda farkı','',
        'Aşağıdaki değerler512×512 girişte Conv2d MAC izidir. Entropy recovery ağları decoder toplamına dahildir; gerçek entropy coder, elementwise işlem ve bellek maliyeti değildir. Hız sonucu olarak okunamaz.',
        '', '| Model | Synthesis GMac | Neural decoder GMac | Encoder+reconstruction GMac | Synthesis azalma % | Neural decoder azalma % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| D{r['depth']} | {r['synthesis_macs']/1e9:.3f} | {r['neural_decoder_macs']/1e9:.3f} | {r['encoder_with_reconstruction_macs']/1e9:.3f} | {r['synthesis_macs_saving_vs_d12_percent']:.2f} | {r['neural_decoder_macs_saving_vs_d12_percent']:.2f} |")
    lines+=['','D8/D10 dahil bütün mimariler sayıldı; bu iki modelin eğitilmiş kalite sonucu yok. Meta-tensor izinin D2/64×64 eşdeğeri gerçek CPU yürütmeyle birebir karşılaştırıldı.',
        'Encoder reconstruction ve entropy coding örtüşebildiğinden, encoder MAC azaltımı da wall-time oranına çevrilemez. MAC değerlerinin toplamsal olması, sürelerin toplamsal olduğu anlamına gelmez.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ('depth','synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs')} for r in rows]))


if __name__=='__main__':main()
