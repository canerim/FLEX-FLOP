"""FUFEXIT1: explicit shared-exit map around a pinned CPU entropy stream.

Research correctness only. Not Microsoft's native wire format or an optimized
decoder. All source-analysis calls are disabled in the decoder worker.
"""
import hashlib,json,os,struct,subprocess,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
SNAPSHOT=ROOT/'research/shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')
REFERENCE=REPO/'experiments/dcvcuf_reference_20260927'
for path in (REFERENCE,SNAPSHOT,UPSTREAM):sys.path.insert(0,str(path))
from reference_codec import ReferenceCodec,parse_container,validate_dimensions,MAX_PAYLOAD

MAGIC=b'FUFEXIT1'
HEADER=struct.Struct('>8sIIII32s32s')
CHECKPOINT=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
RELEASE=ROOT/'reference_d12/released_cvpr2026_image.pth.tar'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pack_map(indices):
    result=bytearray((len(indices)+3)//4)
    for i,k in enumerate(indices):
        if type(k) is not int or not 2<=k<=5:raise ValueError('Only reachable exit indices2..5 are supported')
        result[i//4]|=(k-2)<<(6-2*(i%4))
    return bytes(result)


def wrap(inner,h,w,indices,checkpoint_digest):
    validate_dimensions(h,w)
    count=((h+255)//256)*((w+255)//256)
    if len(indices)!=count:raise ValueError('Map count does not match valid dimensions')
    payload=pack_map(indices)+inner
    return HEADER.pack(MAGIC,h,w,count,len(inner),bytes.fromhex(checkpoint_digest),hashlib.sha256(payload).digest())+payload


def unwrap(data,checkpoint_digest,release_digest):
    if not isinstance(data,bytes) or len(data)<HEADER.size:raise ValueError('Truncated shared stream')
    magic,h,w,count,inner_length,model,checksum=HEADER.unpack_from(data)
    if magic!=MAGIC or model.hex()!=checkpoint_digest:raise ValueError('Unknown stream or reconstruction checkpoint')
    validate_dimensions(h,w)
    expected=((h+255)//256)*((w+255)//256)
    if count!=expected or not 92<=inner_length<=MAX_PAYLOAD+88:raise ValueError('Invalid map or payload dimensions')
    map_bytes=(count+3)//4
    if len(data)!=HEADER.size+map_bytes+inner_length:raise ValueError('Shared stream length mismatch')
    body=data[HEADER.size:]
    if hashlib.sha256(body).digest()!=checksum:raise ValueError('Shared stream checksum mismatch')
    packed=body[:map_bytes]
    if count%4 and packed[-1]&((1<<(2*(4-count%4)))-1):raise ValueError('Nonzero unused map bits')
    indices=[2+((packed[i//4]>>(6-2*(i%4)))&3) for i in range(count)]
    inner=body[map_bytes:];meta=parse_container(inner,12,release_digest)
    if (meta['height'],meta['width'])!=(((h+255)//256)*256,((w+255)//256)*256):raise ValueError('Inner and valid support differ')
    return {'height':h,'width':w,'map':indices,'inner':inner,'map_bytes':map_bytes,'outer_header_bytes':HEADER.size,'inner_header_bytes':meta['header_bytes'],'payload_bytes':meta['payload_bytes'],'qp':meta['qp']}


class Synthesis(torch.nn.Module):
    def __init__(self,decoder):
        super().__init__();self.decoder=decoder;self.exit_map=None
    def set_map(self,indices):self.exit_map=torch.tensor(indices,dtype=torch.long)
    def forward(self,y,q):
        if self.exit_map is None:raise ValueError('Exit map must be supplied by the stream')
        if self.exit_map.numel()!=(y.shape[-2]//16)*(y.shape[-1]//16):raise ValueError('Latent/map geometry differs')
        return self.decoder(y,q,exit_map=self.exit_map)


def load(source_free=False):
    torch.set_num_threads(2);torch.manual_seed(42)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':raise ValueError('Unpinned upstream')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--','src']):raise ValueError('Modified upstream')
    proof=json.loads((REPO/'docs/research/2026-09-27-six-hour/shared_metric_audit/analysis.json').read_text())
    for path,digest in proof['source_sha256'].items():
        if path.startswith(str(SNAPSHOT)) or path.startswith(str(UPSTREAM)):
            if sha(path)!=digest:raise ValueError('Frozen shared implementation changed')
    audit=json.loads((REPO/'cvpr2027/data/refresh20260927/reference_audit.json').read_text())
    for path in (CHECKPOINT,RELEASE):
        key=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        if sha(path)!=audit['hashes'][key]:raise ValueError('Checkpoint identity changed')
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.models.image_model import DMCI
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    shared=FlexUFIntra(FlexUFConfig(**ck['config'])).eval();shared.load_state_dict(ck['state_dict'],strict=True)
    state=torch.load(RELEASE,map_location='cpu',weights_only=False);state=state.get('state_dict',state)
    stock=DMCI().eval();stock.load_state_dict(state,strict=True)
    front={k:v for k,v in state.items() if not k.startswith('dec.')}
    if len(front)!=255 or any(not torch.equal(v,shared.state_dict()[k]) for k,v in front.items()):raise ValueError('Analysis/entropy sharing differs')
    codec=ReferenceCodec(stock,sha(RELEASE),ROOT/'research/reference_entropy_v1')
    synthesis=Synthesis(shared.dec).eval();stock.dec=synthesis
    if source_free:
        def forbidden(*a,**kw):raise AssertionError('Source analysis called inside isolated shared decoder')
        for model in (stock,shared):
            model.enc.forward=forbidden;model.hyper_enc.forward=forbidden
        stock.forward_one_frame=forbidden;shared._encode_to_latent=forbidden
    return codec,synthesis,{'shared_checkpoint_sha256':sha(CHECKPOINT),'released_checkpoint_sha256':sha(RELEASE),'front_tensors_equal':255,'source_analysis_disabled':source_free,'entropy_extension':codec.extension_manifest}


def decode(codec,synthesis,data,identities):
    meta=unwrap(data,identities['shared_checkpoint_sha256'],identities['released_checkpoint_sha256'])
    synthesis.set_map(meta['map']);out,trace=codec.decode(meta['inner'])
    out=out[:,:,:meta['height'],:meta['width']].contiguous()
    return out,trace,{k:v for k,v in meta.items() if k!='inner'}
