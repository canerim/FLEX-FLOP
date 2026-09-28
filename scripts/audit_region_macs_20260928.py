"""Layer-wise rectangular geometry costs for the fixed-map coalescing ablation."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import torch
import torch.nn.functional as F
from audit_depth_macs_20260927 import UPSTREAM,COMMIT
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/dcvcuf_region_merge_20260927'))
from layout import regions,description,checks
OUT=REPO/'docs/research/2026-09-27-six-hour/region_macs'


def trace(depth,height,width,device='meta'):
    from src.models.image_model import DMCI
    layers=[]
    with torch.device(device):
        net=DMCI().eval();net.dec.dec_1=torch.nn.Sequential(*list(net.dec.dec_1.children())[:depth+1])
        def latent_padding(module,args):
            y=args[0];return (F.pad(y,(0,(-y.shape[-1])%4,0,(-y.shape[-2])%4),mode='replicate'),)
        net.hyper_enc.register_forward_pre_hook(latent_padding)
        for name,module in net.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                def count(m,inputs,output,name=name):
                    layers.append({'name':name,'input':list(inputs[0].shape),'output':list(output.shape),
                        'macs':output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]})
                module.register_forward_hook(count)
        with torch.inference_mode():
            result=net.forward_one_frame(torch.zeros(1,3,height,width),torch.tensor([32],dtype=torch.int32),recon_only=True)
        if result.shape!=(1,3,height,width):raise AssertionError('Unexpected reconstruction geometry')
    groups={}
    for r in layers:
        name=r['name'].split('.')[0];groups[name]=groups.get(name,0)+r['macs']
    synthesis=groups['dec'];analysis=groups['enc']+groups['hyper_enc'];total=sum(groups.values())
    return {'depth':depth,'height':height,'width':width,'groups':groups,'layers':layers,
        'synthesis_macs':synthesis,'entropy_neural_macs':total-analysis-synthesis,
        'neural_decoder_macs':total-analysis,'encoder_with_reconstruction_macs':total}


def main():
    torch.set_num_threads(2);torch.manual_seed(20260928)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('Upstream revision differs')
    sys.path.insert(0,str(UPSTREAM));layout=checks();cache={};rows=[]
    for profile in range(10):
        components=[]
        for r in regions(profile):
            t,l,b,w=r['window'];key=(r['depth'],b-t,w-l)
            if key not in cache:cache[key]=trace(*key)
            components.append(cache[key])
        pattern,merged,phase=description(profile)
        rows.append({'profile':profile,'pattern':pattern,'merged':merged,'phase':phase,
            'components':[{k:r[k] for k in ('depth','height','width')} for r in components],
            'image_pixels_coded':sum(r['height']*r['width'] for r in components),
            **{k:sum(r[k] for r in components) for k in ('synthesis_macs','entropy_neural_macs','neural_decoder_macs','encoder_with_reconstruction_macs')}})
    validations=[]
    for shape in ((2,288,512),(2,512,288)):
        real=trace(*shape,device='cpu')
        if real!=cache[shape]:raise AssertionError('Meta trace differs from real CPU Conv2d calls')
        validations.append({'depth':shape[0],'height':shape[1],'width':shape[2],'meta_equals_real_cpu':True})
    baseline=trace(12,512,512);contrasts=[]
    for merged in (4,5,8,9):
        before,after=rows[merged-2],rows[merged]
        contrasts.append({'pattern':after['pattern'],'phase':after['phase'],
            **{k+'_change_percent':100*(after[k]/before[k]-1) for k in ('image_pixels_coded','synthesis_macs','entropy_neural_macs','neural_decoder_macs','encoder_with_reconstruction_macs')}})
    for r in rows:
        for k in ('synthesis_macs','neural_decoder_macs','encoder_with_reconstruction_macs'):r[k+'_percent_of_full_d12']=100*r[k]/baseline[k]
    if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA initialization')
    result={'scope':'Rectangular Conv2d trace with image-pad16/latent-pad4 geometry; no quality, native GPU numerical parity, grouping overhead or wall-time claim.',
        'rows':rows,'paired_merge_changes':contrasts,'component_traces':list(cache.values()),'baseline_full_d12':baseline,
        'real_cpu_checks':validations,'layout_checks':layout,'cuda_initialized':False,'torch':torch.__version__,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),REPO/'experiments/dcvcuf_region_merge_20260927/layout.py',UPSTREAM/'src/models/image_model.py',UPSTREAM/'src/layers/layers.py')},
        'interpretation':'Every profile assigns half the retained source pixels to D2 and half to D6. Coalescing removes context duplicated across an internal boundary without changing pixel depths. Hyperprior dimensions are traced explicitly; costs are not inferred from a single area ratio. No runtime speedup can be inferred.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'contrasts':contrasts,'cpu_checks':validations,'out':str(OUT)}))


if __name__=='__main__':main()
