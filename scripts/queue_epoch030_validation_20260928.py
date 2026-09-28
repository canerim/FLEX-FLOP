"""Wait for fixed epoch30 snapshots, then run the full CPU actual-byte protocol.

Never changes training, uses a GPU, selects a checkpoint by quality, or publishes
unreviewed results. The finite queue expires after48hours if prerequisites fail.
"""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
CODE=REPO/'experiments/dcvcuf_epoch030_reference_20260928'
OUT=ROOT/'research/epoch030_validation_queue'
ANALYSIS=REPO/'scripts/analyze_reference_validation_epoch030_20260928.py'
DOC=REPO/'docs/research/2026-09-27-six-hour'
EPOCH=30;STEP=EPOCH*24049


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())


def ready():
    status={}
    for depth in (2,4,6):
        candidates=[]
        for path in sorted((ROOT/f'runs/d{depth}/validation').glob('step_*.json')):
            value=read(path)
            if value.get('epoch')==EPOCH and value.get('global_step')==STEP:candidates.append(path)
        weights=ROOT/f'runs/d{depth}/weights_epoch{EPOCH:03d}.pt'
        if len(candidates)>1:raise ValueError('Duplicate milestone validation')
        status[str(depth)]={'checkpoint_exists':weights.is_file(),'validation':str(candidates[0]) if candidates else None}
    return all(r['checkpoint_exists'] and r['validation'] for r in status.values()),status


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'progress.json').exists() and read(OUT/'progress.json')['state']=='complete':return
    smoke_path=ROOT/'research/epoch030_pipeline_check_on_epoch020/verification.json'
    smoke=read(smoke_path)
    if not smoke['all_passed'] or len(smoke['cases'])!=36 or smoke['epoch_for_shallow_models']!=20:raise ValueError('Completed pipeline preflight required')
    for name,digest in smoke['code_sha256'].items():
        if sha(CODE/name)!=digest:raise ValueError('Milestone code differs from pipeline preflight')
    files=[Path(__file__),ANALYSIS,*sorted(CODE.glob('*.py'))]
    frozen={str(p):sha(p) for p in files}
    manifest={'scope':__doc__,'epoch':EPOCH,'step':STEP,'images':100,'qps':[0,16,32,48,63],'depths':[2,4,6,12],
        'code_sha256':frozen,'pipeline_preflight_sha256':sha(smoke_path),'selection':'All fixed epoch30 snapshots, regardless of observed quality. ReleasedD12 remains a different-provenance anchor. No training updates.',
        'limit':'Wait at most48hours; then one36-case preflight(max30min), one2000-case full-cohort evaluation(max12h), and analysis(max30min). Stop on failure; no automatic parameter tuning or repeated retries.',
        'resources':'CPU-only, two torch threads, one BLAS thread; launch wrapper with nice10. No native CUDA correctness or timing.'}
    if (OUT/'manifest.json').exists() and read(OUT/'manifest.json')!=manifest:raise ValueError('Queue provenance differs')
    atomic(OUT/'manifest.json',manifest);progress={'state':'waiting_epoch030','pid':os.getpid(),'started_utc':utc(),'target_epoch':EPOCH};atomic(OUT/'progress.json',progress)
    deadline=time.monotonic()+48*3600
    def check_sources():
        for p,digest in frozen.items():
            if sha(p)!=digest:raise ValueError('Queued code changed')
    child=None
    try:
        while True:
            available,models=ready();progress.update(models=models,updated_utc=utc());atomic(OUT/'progress.json',progress)
            if available:break
            if time.monotonic()>=deadline:
                progress.update(state='expired',finished_utc=utc());atomic(OUT/'progress.json',progress);return
            time.sleep(60)
        check_sources()
        # An end-of-epoch validation file is written after the weight snapshot,
        # so readiness does not rely on observing a half-written torch.save file.
        verification=ROOT/'research/reference_verification_epoch030'
        progress.update(state='verifying_epoch030',updated_utc=utc());atomic(OUT/'progress.json',progress)
        with (OUT/'preflight.log').open('w') as log:subprocess.run([sys.executable,str(CODE/'verify_reference.py'),'--epoch','30','--out',str(verification)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO,timeout=1800)
        proof=read(verification/'verification.json')
        if not proof['all_passed'] or len(proof['cases'])!=36 or proof['epoch_for_shallow_models']!=30:raise ValueError('Epoch30 preflight incomplete')
        if any(proof['models'][str(d)]['epoch']!=30 for d in (2,4,6)):raise ValueError('Checkpoint metadata is not epoch30')
        check_sources();evaluation=ROOT/'research/div2k100_reference_epoch030'
        if evaluation.exists():raise ValueError('Refusing to overwrite an existing milestone evaluation')
        progress.update(state='evaluating_epoch030',updated_utc=utc());atomic(OUT/'progress.json',progress)
        with (OUT/'evaluation.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(CODE/'evaluate_validation.py'),'--out',str(evaluation)],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
            manifest_deadline=time.monotonic()+120
            while not (evaluation/'manifest.json').exists():
                if child.poll() is not None:raise RuntimeError('Evaluation exited before manifest')
                if time.monotonic()>manifest_deadline:raise TimeoutError('Evaluation did not produce a startup manifest')
                time.sleep(.25)
            provenance_files=[ROOT/'research/reference_entropy_v1/build_manifest.json',ROOT/'research/reference_entropy_v1/MLCodec_extensions_cpp.cpython-312-x86_64-linux-gnu.so',verification/'verification.json',evaluation/'manifest.json',ROOT/'research/div2k100_center512_rgb/manifest.json']
            provenance={'recorded_utc':utc(),'scope':'Milestone inputs captured at evaluation startup, before outcome analysis. No quality-based checkpoint selection.','files':{str(p):sha(p) for p in provenance_files},'code_sha256':frozen,'queue_manifest_sha256':sha(OUT/'manifest.json')}
            atomic(DOC/'reference_epoch030_provenance.json',provenance)
            code=child.wait(timeout=12*3600)
            if code:raise subprocess.CalledProcessError(code,child.args)
        state=read(evaluation/'progress.json')
        if (state['state'],state['completed'])!=('complete',2000):raise ValueError('Milestone cohort incomplete')
        check_sources();progress.update(state='analyzing_epoch030',updated_utc=utc());atomic(OUT/'progress.json',progress)
        with (OUT/'analysis.log').open('w') as log:subprocess.run([sys.executable,str(ANALYSIS),'--input',str(evaluation),'--out',str(DOC/'div2k100_epoch030')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO,timeout=1800)
        progress.update(state='complete',finished_utc=utc(),completed_cases=2000,report=str(DOC/'div2k100_epoch030/REPORT_TR.md'));atomic(OUT/'progress.json',progress)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=30)
        progress.update(state='failed',error=repr(error),updated_utc=utc());atomic(OUT/'progress.json',progress);raise


if __name__=='__main__':main()
