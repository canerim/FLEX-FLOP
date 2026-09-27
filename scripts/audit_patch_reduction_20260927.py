"""Explain the aborted patch run with one fixed independently decoded reference case."""
import json,math,subprocess,sys,hashlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/dcvcuf_patch_control_20260927'))
from evaluate_patching import ROOT,REFERENCE,checkpoint,receive,isolated,metrics
import numpy as np
import torch

def main():
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'upstream'))
    from src.utils.transforms import ycbcr2rgb
    out=REPO/'docs/research/2026-09-27-six-hour/patch_reduction_fix';out.mkdir(exist_ok=True)
    cases=ROOT/'research/div2k100_reference_epoch020/d2'
    cp=next(c for c in json.loads((ROOT/'research/div2k100_center512_rgb/manifest.json').read_text())['images'] if c['image']=='0801.png')
    rgb=np.load(cp['crop_path'],allow_pickle=False);target=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1)[None].contiguous()
    log=(out/'decoder.stderr.log').open('w');worker=subprocess.Popen([sys.executable,str(REFERENCE/'decode_worker.py'),'--checkpoint',str(checkpoint(2)),'--depth','2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
    try:
        ready=receive(worker);assert ready['ready']
        original=json.loads((cases/'cases/0801_qp16.json').read_text())
        rec,diag=isolated(worker,cases/'streams/0801_qp16.fufref',out/'temporary.npy')
        assert diag['y_hat_sha256']==original['y_hat_sha256']
        x=ycbcr2rgb(rec.clamp(-.5,.5)+.5,clamp=True);squared=(x-target).square()
        direct=float(squared.mean());channel_first=float(squared.mean(1)[0].mean())
        corrected=metrics(x,target)
        assert direct==original['mse_rgb']==corrected['mse_rgb']
        assert corrected['psnr_rgb']==original['psnr_rgb']
        result={'fixed_case':'D2 epoch20 DIV2K0801 QP16','reference_psnr_rgb':original['psnr_rgb'],'old_channel_then_spatial_psnr_rgb':-10*math.log10(channel_first),'corrected_direct_mean_psnr_rgb':corrected['psnr_rgb'],'old_minus_reference_db':-10*math.log10(channel_first)-original['psnr_rgb'],'direct_mean_mse':direct,'channel_then_spatial_mean_mse':channel_first,'latent_hash_matches':True,'corrected_global_metric_exact':True,'scope':'FP32 reduction-order regression; no relaxation of decoder or quality parity tolerances. Original failed directory retained; all patch cases recomputed under the corrected script.'}
        (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
if __name__=='__main__':main()
