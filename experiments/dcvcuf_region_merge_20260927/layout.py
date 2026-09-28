"""Explicit fixed-profile bank framing; geometry is part of the decoder contract."""
from __future__ import annotations
import struct
MAGIC=b'FUFBNK1\x00'
HEADER=struct.Struct('>8sBBH')
LENGTH=struct.Struct('>I')
SIZE=512
HALO=32
MAX_STREAM=64*1024*1024


def description(profile):
    if not isinstance(profile,int) or isinstance(profile,bool) or not 0<=profile<10:raise ValueError('Unknown fixed profile')
    if profile<2:return 'checker',False,profile
    if profile<6:return 'vertical',profile>=4,profile%2
    return 'horizontal',profile>=8,profile%2


def regions(profile):
    pattern,merged,phase=description(profile);rows=[]
    cores=([(0,0,512,256),(0,256,512,512)] if pattern=='vertical' else [(0,0,256,512),(256,0,512,512)]) if merged else [(y,x,y+256,x+256) for y in (0,256) for x in (0,256)]
    for top,left,bottom,right in cores:
        label=((top//256+left//256)%2 if pattern=='checker' else left//256 if pattern=='vertical' else top//256)
        depth=2 if (label+phase)%2==0 else 6
        window=(max(0,top-HALO),max(0,left-HALO),min(SIZE,bottom+HALO),min(SIZE,right+HALO))
        rows.append({'core':(top,left,bottom,right),'window':window,'depth':depth})
    return rows


def pack(profile,parts):
    expected=regions(profile)
    if len(parts)!=len(expected) or any(not isinstance(p,bytes) or not 88<len(p)<=MAX_STREAM for p in parts):raise ValueError('Invalid component count or length')
    return HEADER.pack(MAGIC,profile,len(parts),0)+b''.join(LENGTH.pack(len(p)) for p in parts)+b''.join(parts)


def unpack(data):
    if not isinstance(data,bytes) or len(data)<HEADER.size:raise ValueError('Truncated bank header')
    magic,profile,count,reserved=HEADER.unpack_from(data)
    if magic!=MAGIC or reserved!=0:raise ValueError('Unknown bank format')
    expected=regions(profile)
    if count!=len(expected) or len(data)<HEADER.size+4*count:raise ValueError('Bank component count differs')
    lengths=[LENGTH.unpack_from(data,HEADER.size+4*i)[0] for i in range(count)]
    if any(not 88<n<=MAX_STREAM for n in lengths):raise ValueError('Invalid bank component length')
    offset=HEADER.size+4*count
    if len(data)!=offset+sum(lengths):raise ValueError('Truncated or trailing bank bytes')
    parts=[]
    for n in lengths:parts.append(data[offset:offset+n]);offset+=n
    return profile,list(zip(expected,parts))


def checks():
    import numpy as np
    counts=[]
    for profile in range(10):
        cover=np.zeros((512,512),dtype=np.int8);depths=np.zeros_like(cover)
        for r in regions(profile):
            t,l,b,e=r['core'];wt,wl,wb,we=r['window'];cover[t:b,l:e]+=1;depths[t:b,l:e]=r['depth']
            assert 0<=wt<=t<b<=wb<=512 and 0<=wl<=l<e<=we<=512
        assert np.all(cover==1) and np.sum(depths==2)==512**2//2 and np.sum(depths==6)==512**2//2
        payload=pack(profile,[b'x'*100 for r in regions(profile)]);p,parts=unpack(payload);assert p==profile and len(parts)==len(regions(profile));counts.append(len(parts))
        if profile in (4,5,8,9):
            before=np.zeros_like(depths)
            for r in regions(profile-2):t,l,b,e=r['core'];before[t:b,l:e]=r['depth']
            assert np.array_equal(before,depths)
    invalid=[b'',b'x'*12,pack(0,[b'x'*100]*4)[:-1],pack(0,[b'x'*100]*4)+b'x',HEADER.pack(MAGIC,10,4,0),HEADER.pack(MAGIC,0,2,0),HEADER.pack(MAGIC,0,4,1)]
    for p in invalid:
        try:unpack(p)
        except ValueError:continue
        raise AssertionError('Malformed bank accepted')
    return {'profiles':10,'core_coverage_exact':True,'equal_depth_pixel_counts':True,'merged_maps_unchanged':True,'component_counts':counts,'malformed_containers_rejected':len(invalid)}
if __name__=='__main__':
    import json
    print(json.dumps(checks()))
