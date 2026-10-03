"""Load a pinned DCVC-UF model on CPU without touching training processes."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
UPSTREAM=ROOT/'upstream'
COMMIT='cbdae87a5445114cdc7f48816da63ea80bdeac40'


def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load_model(path,depth,upstream=UPSTREAM):
    if depth not in (2,4,6,8,10,12):raise ValueError('Unsupported depth')
    upstream=Path(upstream).resolve();path=Path(path).resolve()
    head=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError('Unpinned upstream model')
    if subprocess.check_output(['git','-C',str(upstream),'diff','--','src/models','src/layers/layers.py']):
        raise ValueError('Upstream model sources have local changes')
    sys.path.insert(0,str(upstream))
    from src.models.image_model import DMCI
    before=file_sha(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if 'depth' in checkpoint and checkpoint['depth']!=depth:
        raise ValueError('Checkpoint depth differs from requested depth')
    net=DMCI()
    net.dec.dec_1=torch.nn.Sequential(*list(net.dec.dec_1.children())[:depth+1])
    state=checkpoint.get('state_dict',checkpoint.get('net',checkpoint))
    net.load_state_dict(state,strict=True)
    if any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Non-finite or non-FP32 parameters')
    if file_sha(path)!=before:raise RuntimeError('Checkpoint changed while loading')
    net.eval()
    record={'path':str(path),'sha256':before,'depth':depth,'upstream_commit':COMMIT,
            'parameters':sum(p.numel() for p in net.parameters()),
            'epoch':checkpoint.get('epoch'),'global_step':checkpoint.get('global_step'),
            'model_source_sha256':file_sha(upstream/'src/models/image_model.py'),
            'common_source_sha256':file_sha(upstream/'src/models/common_model.py')}
    return net,record
