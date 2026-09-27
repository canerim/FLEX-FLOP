"""Bound patch geometry costs and verify the encoder's structural support."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from audit_depth_macs_20260927 import ROOT,UPSTREAM,COMMIT,trace

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'docs/research/2026-09-27-six-hour/patch_geometry'


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
    rows=[]
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
    checks=[]
    for depth in (2,12):
        full=next(r for r in mac['rows'] if r['depth']==depth)
        for side in (256,320):
            observed=trace(depth,side,side)
            for key in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):
                assert observed[key]*512**2==full[key]*side**2
            checks.append({'depth':depth,'side':side,'meta_trace_matches_area_scaling':True})
    files=[Path(__file__),mp,UPSTREAM/'src/models/image_model.py',UPSTREAM/'src/layers/layers.py']
    result={'scope':'Architecture/geometry audit, not measured latency, quality, effective receptive field or a context-sufficiency guarantee',
        'encoder_support':{'latent_position_1d_formula':'[16*i-64,16*i+71] inclusive','width_pixels':136,
            'method':'Count7 depthwise3x3 spatial blocks after pixel_unshuffle8, then3x3 stride2 convolution; verify random-weight real-CPU input gradient bounding box',
            'gradient_probe_latent_position':[index,index],'input_shape':[192,192],'observed_bbox_ymin_ymax_xmin_xmax':bounding,
            'limitation':'Only the analysis transform. Hyperprior, autoregressive spatial context and synthesis introduce additional dependencies. Nonzero gradient support in one random-weight probe is a structural check, not a trained effective receptive-field estimate.'},
        'rows':rows,'trace_checks':checks,'upstream_commit':COMMIT,'torch':torch.__version__,'cuda_initialized':False,
        'cost_scope':'Conv2d MACs only. Four independent patch calls can add launch/rANS/transfer/initialization overhead; these are excluded. Native encoder reconstruction and entropy coding overlap, so MAC ratios are not wall-time ratios.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Halo, padding ve depth tasarrufunun paydası','',
        '2×2 gridde her256 core yalnız bir iç sınır yönünde context alır. Halo32:288→320 padding; halo64:320→320. İkisinde de kodlanan alan full512\'nin1,5625 katı.',
        'Halo64 pencere başlangıcını64-pixel hyperlatent gridine de hizalar. Bu eşit PSNR, payload veya latency anlamına gelmez.',
        '', '| Model | Halo | Neural decoder / full D12 (%) | Encoder+reconstruction / full D12 (%) |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| D{r['depth']} | {r['halo']} | {r['neural_decoder_macs_percent_of_full_d12']:.2f} | {r['encoder_with_reconstruction_macs_percent_of_full_d12']:.2f} |")
    lines += ['', 'Bu tablo kalite eşleştirmesi değil, aynı source alanını kodlayan mimarilerin Conv2d hesabıdır. D8/D10 eğitilmiş sonuç değildir. Entropy coder, çağrı sayısı, bellek ve MLP maliyetleri yok.',
        '', 'Encoder latentinin doğrudan yapısal source aralığı136 piksel: [16i−64,16i+71]. Kernel incelemesi,192×192 gerçekCPU gradient probe\'unda bbox[32,167] ile doğrulandı. Halo32 tek başına full-frame analiz eşdeğerliğini garanti etmez. Tüm codec receptive field veya gerekli minimum halo bundan ibaret değildir.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'encoder_support':result['encoder_support'],'trace_checks':checks,'out':str(OUT)}))


if __name__=='__main__':main()
