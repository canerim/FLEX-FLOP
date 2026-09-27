"""Bound patch geometry costs and verify the encoder's structural support."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
from audit_depth_macs_20260927 import ROOT,UPSTREAM,COMMIT,trace

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'docs/research/2026-09-27-six-hour/patch_geometry'


def native_shape_trace(depth,side,device='meta'):
    from src.models.image_model import DMCI
    layers=[]
    with torch.device(device):
        net=DMCI().eval();net.dec.dec_1=torch.nn.Sequential(*list(net.dec.dec_1.children())[:depth+1])
        def pad_latent(module,args):
            y=args[0]
            return (F.pad(y,(0,(-y.shape[-1])%4,0,(-y.shape[-2])%4),mode='replicate'),)
        net.hyper_enc.register_forward_pre_hook(pad_latent)
        for name,module in net.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                def count(m,inputs,out,name=name):
                    layers.append({'name':name,'input':list(inputs[0].shape),'output':list(out.shape),
                        'macs':out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]})
                module.register_forward_hook(count)
        with torch.inference_mode():
            rec=net.forward_one_frame(torch.zeros(1,3,side,side),torch.tensor([32],dtype=torch.int32),recon_only=True)
        assert rec.shape==(1,3,side,side)
    groups={}
    for r in layers:
        group=r['name'].split('.')[0];groups[group]=groups.get(group,0)+r['macs']
    return {'groups':groups,'layers':layers}


def main():
    torch.set_num_threads(2);torch.manual_seed(20260927)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise ValueError('Upstream revision changed')
    sys.path.insert(0,str(UPSTREAM))
    from src.models.image_model import IntraEncoder
    from src.layers.layers import DepthConvBlock
    encoder=IntraEncoder().eval()
    blocks=[m for m in encoder.modules() if isinstance(m,DepthConvBlock)]
    assert len(blocks)==7
    for block in blocks:
        spatial=[m for m in block.modules() if isinstance(m,torch.nn.Conv2d) and m.kernel_size!=(1,1)]
        assert len(spatial)==1 and spatial[0].kernel_size==(3,3) and spatial[0].stride==(1,1)
    final=encoder.enc_2[-1]
    assert final.kernel_size==(3,3) and final.stride==(2,2) and final.padding==(1,1)
    # Each pixel-unshuffle group covers8 inputs; seven spatial blocks expand
    # seven feature positions in either direction; final stride2 convolution
    # adds one. At latent position i this gives[16i-64,16i+71].
    index=6;expected=[16*index-64,16*index+71]
    x=(torch.randn(1,3,192,192)*.1).requires_grad_()
    output=encoder(x,torch.ones((1,384,1,1)))
    output[:,:,index,index].sum().backward()
    support=(x.grad.abs().sum((0,1))>0).nonzero()
    bounding=[int(support[:,0].min()),int(support[:,0].max()),int(support[:,1].min()),int(support[:,1].max())]
    assert bounding==expected*2
    assert not torch.cuda.is_initialized()
    mp=REPO/'docs/research/2026-09-27-six-hour/depth_macs/analysis.json';mac=json.loads(mp.read_text())
    baseline=next(r for r in mac['rows'] if r['depth']==12)
    rows=[];native_rows=[]
    for entry in mac['rows']:
        for halo in (0,32,64):
            side=256+halo;padded=(side+63)//64*64;area_ratio=4*padded**2/512**2
            r={'depth':entry['depth'],'halo':halo,'context_window_side':side,'padded_side':padded,
               'coded_area_ratio':area_ratio,'window_start_coordinates':[0,256-halo],
               'all_starts_aligned_to_hyperlatent64':(256-halo)%64==0}
            for key in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):
                value=entry[key]*area_ratio
                r[key]=int(value);r[key+'_percent_of_full_d12']=100*value/baseline[key]
            rows.append(r)
            native=native_shape_trace(entry['depth'],side)
            groups=native['groups'];dec=groups['dec'];analysis=groups['enc']+groups['hyper_enc']
            recovery=sum(groups.values())-analysis-dec
            row={'depth':entry['depth'],'halo':halo,'image_side':side,'latent_side':side//16,
                 'hyperlatent_side':(side//16+3)//4,'groups_per_patch':groups}
            for key,value in [('synthesis_macs',dec),('neural_decoder_macs',dec+recovery),
                              ('encoder_with_reconstruction_macs',sum(groups.values()))]:
                row[key]=4*value;row[key+'_percent_of_full_d12']=100*4*value/baseline[key]
            native_rows.append(row)
    checks=[]
    for depth in (2,12):
        full=next(r for r in mac['rows'] if r['depth']==depth)
        for side in (256,320):
            observed=trace(depth,side,side)
            for key in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):
                assert observed[key]*512**2==full[key]*side**2
            checks.append({'depth':depth,'side':side,'meta_trace_matches_area_scaling':True})
    native_meta=native_shape_trace(2,288);native_cpu=native_shape_trace(2,288,'cpu')
    assert native_meta==native_cpu
    checks.append({'depth':2,'side':288,'native_geometry_meta_matches_real_cpu_conv_trace':True})
    for row in native_rows:
        if row['halo'] in (0,64):
            other=next(r for r in rows if (r['depth'],r['halo'])==(row['depth'],row['halo']))
            for key in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):assert row[key]==other[key]
    files=[Path(__file__),mp,UPSTREAM/'src/models/image_model.py',UPSTREAM/'src/layers/layers.py',
           UPSTREAM/'test_video.py',UPSTREAM/'src/layers/extensions/inference/dmci_proxy.cpp',
           UPSTREAM/'src/layers/extensions/inference/dmc_common.cpp']
    result={'scope':'Architecture/geometry audit, not measured latency, quality, effective receptive field or a context-sufficiency guarantee',
        'encoder_support':{'latent_position_1d_formula':'[16*i-64,16*i+71] inclusive','width_pixels':136,
            'method':'Count7 depthwise3x3 spatial blocks after pixel_unshuffle8, then3x3 stride2 convolution; verify random-weight real-CPU input gradient bounding box',
            'gradient_probe_latent_position':[index,index],'input_shape':[192,192],'observed_bbox_ymin_ymax_xmin_xmax':bounding,
            'limitation':'Only the analysis transform. Hyperprior, autoregressive spatial context and synthesis introduce additional dependencies. Nonzero gradient support in one random-weight probe is a structural check, not a trained effective receptive-field estimate.'},
        'rows':rows,'pad64_scope':'These rows describe theFUFREF1 CPU image-pad64 geometry only; native CUDA pads the image to16 and the hyperanalysis latent separately to4.',
        'native_geometry_rows':native_rows,'native_geometry_scope':'Conv2d shapes replayed with image-pad16 and latent-pad4; no CUDA kernels executed. A realCPU D2/288 trace matches the meta trace. FP32 reference and native FP16 numerical paths may differ.',
        'trace_checks':checks,'upstream_commit':COMMIT,'torch':torch.__version__,'cuda_initialized':False,
        'cost_scope':'Conv2d MACs only. Four independent patch calls can add launch/rANS/transfer/initialization overhead; these are excluded. Native encoder reconstruction and entropy coding overlap, so MAC ratios are not wall-time ratios.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Halo, padding ve depth tasarrufunun paydası','',
        'Aşağıdaki ilk tablo yalnız FUFREF1 CPU image-pad64 protokolüne aittir. Native CUDA image-pad16 + latent-pad4 uygular; aynı maliyet olduğu varsayılamaz.',
        '2×2 gridde her256 core yalnız bir iç sınır yönünde context alır. CPU-pad64 için halo32:288→320 padding; halo64:320→320. İkisinde de kodlanan alan full512\'nin1,5625 katı.',
        'Halo64 pencere başlangıcını64-pixel hyperlatent gridine de hizalar. Bu eşit PSNR, payload veya latency anlamına gelmez.',
        '', '| Model | Halo | Neural decoder / full D12 (%) | Encoder+reconstruction / full D12 (%) |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| D{r['depth']} | {r['halo']} | {r['neural_decoder_macs_percent_of_full_d12']:.2f} | {r['encoder_with_reconstruction_macs_percent_of_full_d12']:.2f} |")
    lines += ['', '## Native boyut düzeniyle ayrı Conv2d hesabı','',
        '288 görüntü native yolda288 kalır; y18→20 yalnız hyperanalysis için pad edilir. Hyperprior/fusion ve diğer ağların alanları aynı oranda büyümediğinden tek image-area çarpanı kullanılamaz. Aşağıdaki değerler katman bazında yeniden izlendi.',
        '', '| Model | Halo | Neural decoder / full D12 (%) | Encoder+reconstruction / full D12 (%) |',
        '|---|---:|---:|---:|']
    for r in native_rows:
        lines.append(f"| D{r['depth']} | {r['halo']} | {r['neural_decoder_macs_percent_of_full_d12']:.2f} | {r['encoder_with_reconstruction_macs_percent_of_full_d12']:.2f} |")
    lines += ['', 'Bu tablo kalite eşleştirmesi değil, aynı source alanını kodlayan mimarilerin Conv2d hesabıdır. D8/D10 eğitilmiş sonuç değildir. Entropy coder, çağrı sayısı, bellek ve MLP maliyetleri yok.',
        '', 'Encoder latentinin doğrudan yapısal source aralığı136 piksel: [16i−64,16i+71]. Kernel incelemesi,192×192 gerçekCPU gradient probe\'unda bbox[32,167] ile doğrulandı. Halo32 tek başına full-frame analiz eşdeğerliğini garanti etmez. Tüm codec receptive field veya gerekli minimum halo bundan ibaret değildir.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'encoder_support':result['encoder_support'],'trace_checks':checks,'out':str(OUT)}))


if __name__=='__main__':main()
