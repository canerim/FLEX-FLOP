"""Exact-data execution geometry, measured 3D trajectories and source-map atlas."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection


def budget(F,d,book):
    fig=plt.figure(figsize=(183*F.MM,72*F.MM))
    ax=fig.add_axes([.035,.19,.49,.72],projection='3d',proj_type='ortho')
    ax.view_init(elev=27,azim=-62)
    ax.set_box_aspect((1.5,1,.85))
    rules=['uniform','dither','router','oracle']
    for i,r in enumerate(rules):
        rows=[v for v in d['summary'] if v['rule']==r]
        ys=[v['mean'] for v in rows]
        verts=[(F.BUD[0],0),*zip(F.BUD,ys),(F.BUD[-1],0)]
        pc=PolyCollection([verts],facecolor=F.COL[r],edgecolor='none',alpha=.11)
        ax.add_collection3d(pc,zs=i,zdir='y')
        ax.plot(F.BUD,[i]*6,ys,color=F.COL[r],lw=1.5,marker=F.MARK[r],ms=4,mec='white',mew=.4)
    ax.set(xlim=(.04,.51),ylim=(-.2,3.2),zlim=(0,42),zticks=[0,20,40])
    ax.set_xticks([.05,.2,.5],['0.05','0.20','0.50'])
    ax.set_yticks(range(4),['Uniform','Dither','Router','Search'])
    ax.tick_params(pad=0,labelsize=7)
    ax.tick_params(axis='y',pad=5)
    ax.tick_params(axis='x',pad=1)
    ax.set_xlabel('Nominal loss target (dB)',labelpad=3)
    ax.set_zlabel('MAC saving (%)',labelpad=2,fontsize=7.5)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):
        a.pane.set_facecolor((.97,.98,.98,1));a.pane.set_edgecolor('white')
        a._axinfo['grid'].update(color=F.GRID,linewidth=.4)
    ax.text2D(.0,1.04,'a',transform=ax.transAxes,weight='bold',fontsize=9)
    ax.text2D(.07,1.04,'Budget shapes the compute frontier',transform=ax.transAxes,fontsize=8)
    ax=fig.add_axes([.665,.25,.315,.59]);F.panel(ax,'b','Increment over dithering')
    ax.grid(False);ax.grid(axis='x',color=F.GRID,lw=.4)
    ax.axhspan(.55,1.45,color='#EDF5F3',zorder=0)
    for name,c,m,dy in [('oracle_minus_dither',F.TEAL,'o',-.13),('router_minus_dither',F.BLUE,'s',.13)]:
        rows=[r for r in d['contrasts'] if r['contrast']==name]
        v=np.array([r['mean'] for r in rows]);lo=np.array([r['lo'] for r in rows]);hi=np.array([r['hi'] for r in rows])
        ax.errorbar(v,np.arange(6)+dy,xerr=[v-lo,hi-v],fmt=m,color=c,ms=4,mec='white',mew=.4,lw=1,capsize=2)
    ax.axvline(0,color=F.MUTED,lw=.6)
    ax.set_yticks(range(6),['0.05','0.10','0.15','0.20','0.30','0.50'])
    ax.set(xlim=(-.35,7.4),ylim=(5.55,-.55),xticks=[0,2,4,6],xlabel='Extra MAC saving (percentage points)',ylabel='Nominal target (dB)')
    handles=[Line2D([],[],color=F.COL[k],marker=F.MARK[k],lw=1.2,ms=4,label=F.LABEL[k]) for k in rules]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=4,fontsize=7,handlelength=1.3,columnspacing=1.3)
    F.audit_and_save(fig,'fig2_budget_value','a: Orthographic 3D display of six measured nominal-budget operating points per policy. Policy is a categorical axis; translucent curtains and segments are guides, not an interpolated surface. b: paired search-minus-dither and router-minus-dither means with 95% sequence-cluster intervals. Common cohort counts 204,263,265,265,265,265. Source-calibrated controls; nominal targets are not final-output guarantees. Synthesis MAC saving excludes routing, entropy and signalling; not latency.',book)


def execution(F,d,book):
    ex=d['example'];m=np.asarray(ex['rules']['router']['0.1']['map']).reshape(ex['grid']);depth=2*(m+1)
    ns=[int((depth>=k).sum()) for k in [6,8,10,12]]
    assert ns==[40,27,7,3]
    fig=plt.figure(figsize=(89*F.MM,66*F.MM))
    ax=fig.add_axes([0,0,1,1],xlim=(0,89),ylim=(0,73));ax.axis('off')
    F.text(ax,3,69,'Spatial work disappears as tiles exit',fontsize=8,weight='bold')
    F.text(ax,3,64,'Recorded 5 × 8 tile grid · one shared latent',fontsize=7,color=F.MUTED)
    # Oblique feature planes preserve the measured row/column coordinates.
    def plane(x,y,stage=None,scale=1):
        def p(c,r):return (x+scale*(c*2.7+r*.65),y+scale*(r*1.15-c*.19))
        hull=[p(0,0),p(8,0),p(8,5),p(0,5)]
        for shift in [1.3,.65]:
            ax.add_patch(Polygon([(xx,yy-shift) for xx,yy in hull],fc='#DCE5E9',ec='white',lw=.45,zorder=1))
        for r in range(5):
            for c in range(8):
                val=depth[4-r,c];live=stage is None or val>=stage
                col=F.DEPTH[(int(val)-6)//2] if stage is None else F.DEPTH[(stage-6)//2] if live else '#F2F5F6'
                ax.add_patch(Polygon([p(c,r),p(c+1,r),p(c+1,r+1),p(c,r+1)],fc=col,ec='white',lw=.5,zorder=3))
    for i,k in enumerate([6,8,10,12]):
        y=51-i*12
        plane(18,y,k)
        F.text(ax,3,y+2,f'{k}',fontsize=14,weight='bold',color=F.DEPTH[i])
        F.text(ax,3,y-2.6,'blocks',fontsize=7,color=F.MUTED)
        F.text(ax,47,y+2,f'{ns[i]}',fontsize=12,weight='bold',color=F.DEPTH[i])
        F.text(ax,47,y-2.3,'active',fontsize=7,color=F.MUTED)
        if i<3:F.arrow(ax,(28,y-2.4),(28,y-6.1),color=F.DEPTH[i],lw=1.2)
        leave=int((depth==k).sum())
        F.arrow(ax,(57,y+2),(64,y+2),color=F.DEPTH[i],lw=1)
        # Small layered adapter glyph; routes converge on one canvas bus.
        if k<12:
            F.strip(ax,65,y-.5,2,w=1.2,h=4.5,color=F.DEPTH[i])
        else:
            ax.plot([65,69],[y+1.4,y+1.4],color=F.DEPTH[i],lw=1.2)
            F.text(ax,67,y-1.8,'I',ha='center',fontsize=7,color=F.DEPTH[i])
        F.text(ax,76,y+3.2,f'{leave} exit',ha='center',fontsize=7,color=F.DEPTH[i])
        ax.plot([70,84,84],[y+1.4,y+1.4,10],color=F.DEPTH[i],lw=.7,alpha=.8)
    F.text(ax,3,5,'77 / 160 tile–block-pair evaluations',fontsize=7.5,weight='bold')
    F.text(ax,3,1.7,'Counts describe execution; they are not timings.',fontsize=7,color=F.MUTED)
    F.arrow(ax,(84,10),(67,10),color=F.INK)
    F.text(ax,65,10,'Shared canvas',ha='right',fontsize=7,color=F.INK)
    F.audit_and_save(fig,'fig8_active_tiles','Measured-map spatial execution for videoSRC05, QP32, nominal 0.1 dB source-calibrated control. Oblique feature planes retain the 5-by-8 tile coordinates: active populations 40/27/7/3 at total depths 6/8/10/12. Departures 13/20/4/3 pass through exit adapters to the shared canvas; I denotes the identity at depth12. Pale positions no longer execute. The four suffix block pairs perform 77 tile-pair evaluations rather than 160 for a dense 12-block pass. This excludes stem/head/adapter cost and is not a GPU timing or batching benchmark.',book)


def gallery(F,d,book):
    root=F.ROOT if F.BUNDLED else F.ROOT/'cvpr2027'
    data=json.loads((root/'data/gallery20260928/gallery.json').read_text());thumbs=np.load(root/'data/gallery20260928/source_luma.npz')
    fig,ax=F.schematic(70)
    rows=data['rows'];heads=['Source','R · .05','R · .10','R · .30','D · .10','S · .10']
    for block in range(2):
        x0=2+block*93
        for j,h in enumerate(heads):F.text(ax,x0+j*14.5+6.7,66,h,ha='center',fontsize=7,weight='bold')
        for rr in range(4):
            idx=block*4+rr;r=rows[idx];H,W=r['hw'];y=50-rr*13.3
            name=r['seq'].split('_')[0]
            F.text(ax,x0,y+10.4,chr(97+idx),weight='bold',fontsize=8)
            F.text(ax,x0+4,y+10.4,name+' · '+r['cls'],fontsize=7,color=F.MUTED)
            for j,v in enumerate(data['views']):
                x=x0+j*14.5;w=13.3;h=w*H/W
                if v['policy']=='source':
                    ax.imshow(thumbs[str(idx)],cmap='gray',vmin=0,vmax=255,extent=(x,x+w,y,y+h),aspect='auto',interpolation='lanczos')
                else:
                    item=r['rules'][v['policy']][str(v['budget'])]
                    if item is None:
                        ax.add_patch(Rectangle((x,y),w,h,fc='#F2F5F6',ec=F.GRID,lw=.4));F.text(ax,x+w/2,y+h/2,'N/A',ha='center',color=F.MUTED,fontsize=7)
                        continue
                    m=np.asarray(item['map']).reshape(r['grid'])
                    for iy in range(m.shape[0]):
                        for ix in range(m.shape[1]):
                            l,ri=ix*256,min((ix+1)*256,W);t,b=iy*256,min((iy+1)*256,H)
                            if ri<=l or b<=t:continue
                            ax.add_patch(Rectangle((x+l/W*w,y+(H-b)/W*w),(ri-l)/W*w,(b-t)/W*w,fc=F.DEPTH[int(m[iy,ix])-2],ec='white',lw=.22))
                ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='#BCC8CD',lw=.35))
    for i,k in enumerate([6,8,10,12]):
        x=3+i*12;ax.add_patch(Rectangle((x,3),2.4,2.4,fc=F.DEPTH[i],ec='none'));F.text(ax,x+3.5,4,str(k),fontsize=7)
    F.text(ax,52,4,'blocks',fontsize=7,color=F.MUTED)
    F.text(ax,75,4,'R: router   D: dither   S: source search · QP32',fontsize=7,color=F.MUTED)
    F.audit_and_save(fig,'fig5_spatial_decisions','Eight sources, six views each: source luma; router nominal targets 0.05/0.10/0.30 dB; Bayer dither and source-informed search at 0.10 dB. a–h are source groups, selected by fixed class quotas and sequence-name hash before inspecting outcomes. Recorded QP32 maps, cropped to valid frame extent; missing/infeasible maps retained as N/A. Luma images supply spatial context, not reconstructions or evidence of visual quality. Colours show 6/8/10/12 executed blocks. All controls here are source-calibrated.',book)


def delivered(F,book):
    d=json.loads((F.ROOT/'data/refresh20260927/delivered_frontier_audit.json').read_text())
    fig,axs=plt.subplots(2,1,figsize=(89*F.MM,78*F.MM))
    fig.subplots_adjust(left=.22,right=.97,bottom=.135,top=.92,hspace=.75)
    ax=axs[0];F.panel(ax,'a','Achieved outputs · 0.10 dB cap')
    for r in [r for r in d['summary'] if r['cap']==.1]:
        key=r['rule'];x=r['mean_loss'];y=r['mean'];ax.scatter(x,y,color=F.COL[key],marker=F.MARK[key],s=28,edgecolor='white',lw=.4,zorder=3)
        offsets={'uniform':(5,-2),'dither':(-5,-8),'router':(-5,0),'oracle':(-5,7)}
        dx,dy=offsets[key];ax.annotate({'uniform':'Uniform','dither':'Dither','router':'Router','oracle':'Search'}[key],(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>0 else 'right',fontsize=7,color=F.COL[key])
    ax.set(xlim=(.063,.084),ylim=(21.5,30.8),xticks=[.065,.075,.080],yticks=[22,26,30],xlabel='Mean achieved 444 loss (dB)',ylabel='MAC saved (%)')
    ax=axs[1];F.panel(ax,'b','Router increment over dithering')
    ax.grid(False);ax.grid(axis='x',color=F.GRID,lw=.4);ax.axvline(0,color=F.MUTED,lw=.6);ax.axhspan(.6,1.4,color='#EDF5F3')
    rows=[r for r in d['contrasts'] if r['contrast']=='router_minus_dither'];v=np.array([r['mean'] for r in rows]);lo=np.array([r['lo'] for r in rows]);hi=np.array([r['hi'] for r in rows])
    ax.errorbar(v,np.arange(6),xerr=[v-lo,hi-v],fmt='o',color=F.BLUE,ms=4,lw=1,capsize=2)
    ax.set_yticks(range(6),[f"{r['cap']:.2f}" for r in rows]);ax.set(xlim=(-.3,4.3),ylim=(5.6,-.6),xticks=[0,1,2,3,4],xlabel='Extra MAC saving (percentage points)',ylabel='Delivered cap (dB)')
    F.audit_and_save(fig,'figS7_delivered_increment','a: All four policies under a common 0.10 dB delivered 444-loss cap; axes show actual mean loss and modelled synthesis MAC saving. Fallback counts uniform/dither/router/search: 11/2/1/2 of265. b: Router-minus-dither paired saving, 95% sequence-cluster intervals across six caps, all265 frame-QP pairs retained. Source-aware retrospective candidate selection, not autonomous control or runtime.',book)
