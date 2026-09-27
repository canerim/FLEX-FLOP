"""Vector research figures for the protocol supplement and codec-bank plan.

All future system paths are explicitly schematic. Numeric diagnostics consume
CPU audits of frozen tables; no new reconstruction or timing is implied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Polygon, Circle
import numpy as np

ROOT, DATA = F.ROOT, F.DATA
F.OUT = ROOT / ("figs/extended20260927" if F.BUNDLED else "paper/figures/extended20260927")
T, A, H = F.text, F.arrow, F.heading


def network(ax, x, y):
    widths = [3, 4, 3]
    points = [[(x + layer * 5, y + (i - (n - 1) / 2) * 3.2) for i in range(n)]
              for layer, n in enumerate(widths)]
    for left, right in zip(points[:-1], points[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#c0cbd2", lw=.35)
    for layer in points:
        for xx, yy in layer:
            ax.add_patch(Circle((xx, yy), .85, fc=F.BLUE, ec="white", lw=.35, zorder=3))


def bank(book):
    fig, ax = F.schematic(142)
    H(ax, 3, 137, "a", "Independent codecs: depth changes the synthesis network, training changes the whole codec")
    xs, depths = [13, 44, 75, 106, 137, 168], [2, 4, 6, 8, 10, 12]
    for x, depth in zip(xs, depths):
        planned = depth in [8, 10]
        c = F.GREY if planned else F.ORANGE if depth == 12 else F.BLUE
        T(ax, x, 129, f"D{depth}", ha="center", fontsize=8, weight="bold", color=c)
        # An hourglass glyph gives each complete analysis/synthesis pair its
        # own latent bottleneck; all dimensions here are schematic.
        ax.add_patch(Polygon([[x-8,121],[x+8,121],[x+4,115],[x-4,115]],
                             fc="white" if planned else c, alpha=1 if planned else .22, ec=c,
                             hatch="///" if planned else None, lw=.6))
        for i in range(4):
            ax.add_patch(Rectangle((x - 3.7 + i*2, 110), 1.5, 2.5, fc=c, ec="white", lw=.3))
        ax.add_patch(Polygon([[x-4,107],[x+4,107],[x+8,101],[x-8,101]],
                             fc="white", ec=c, hatch="///" if planned else None, lw=.6))
        for i in range(depth):
            ax.add_patch(Rectangle((x - 8 + i * 1.35, 96), 1.0, 3.5, fc=c, ec="white", lw=.2))
        T(ax, x, 92, f"{depth} synthesis blocks", ha="center", fontsize=5.8)
        T(ax, x, 86, "PLANNED" if planned else "RELEASED" if depth == 12 else "TRAINING", ha="center",
          color=c, weight="bold", fontsize=5.7)
    T(ax, 5, 77, "D2 / D4 / D6: pinned official recipe, all codec weights learned jointly", fontsize=6)
    T(ax, 5, 72, "A same-recipe D12 control is still needed to separate depth from training provenance.", fontsize=6)
    ax.plot([3, 180], [67, 67], color=F.GRID, lw=.5)

    H(ax, 3, 62, "b", "Proposed deployment: choose before encoding, group work by expert, transmit the choice")
    # A small image plane, neural graph and batch stacks show physical objects,
    # while the stream is a segment tape rather than another module rectangle.
    F.tensor(ax, 5, 38, 15, 12, grid=3)
    T(ax, 12.5, 33, "RGB patches", ha="center", fontsize=6)
    A(ax, (23,44), (33,44))
    network(ax, 37, 44)
    T(ax, 42, 33, "Cheap MLP", ha="center", fontsize=6)
    A(ax, (51,44), (61,44))
    for i, (c, n) in enumerate([(F.BLUE,4),(F.TEAL,3),(F.ORANGE,2)]):
        for j in reversed(range(n)):
            ax.add_patch(Rectangle((65+i*6+j*.4,40+j*.7),4.3,7,fc=c,ec="white",lw=.4))
    T(ax, 73, 33, "Group by expert", ha="center", fontsize=6)
    A(ax, (83,44), (93,44))
    F.strip(ax, 98, 41, 6, w=1.5, h=6, color=F.BLUE)
    T(ax, 104, 33, "Selected codec", ha="center", fontsize=6)
    A(ax, (113,44), (123,44))
    for i, (w, c, label) in enumerate([(5,F.ORANGE,"ID"),(10,F.BLUE,"z"),(23,F.TEAL,"y")]):
        x = [126,131,141][i]
        ax.add_patch(Rectangle((x,40),w,8,fc=c,ec="white",lw=.5))
        T(ax,x+w/2,44,label,ha="center",color="white",fontsize=6)
    T(ax, 145, 33, "Control + coded payloads", ha="center", fontsize=6)
    T(ax, 5, 25, "The decoder reads the expert ID and uses that expert's entropy model and synthesis transform.", fontsize=6)
    ax.plot([3, 180], [19, 19], color=F.GRID, lw=.5)
    T(ax, 5, 13, "Measure: full bytes · encoder + decoder wall time · final cropped quality · bank memory", weight="bold", fontsize=6)
    T(ax, 5, 6, "Deployment is a proposed design. The glyphs and queues are schematic; they do not show measured routes or throughput.", fontsize=6)
    F.audit_and_save(fig,"figS4_codec_bank_plan",
        "Independent depth-reduced codecs and a proposed encoder-side routing path. D2, D4 and D6 are training under the pinned official recipe; D8 and D10 are planned; D12 is the released reference. An equally trained D12 control is required for a depth-only conclusion. Each expert owns its analysis transform, entropy model and synthesis transform. The proposed MLP selects an expert using information available before that expert is encoded. Grouped execution, control signalling and entropy state resets must be included in measurement. Network glyphs, block strips, queue sizes and stream segments are schematic, with no quantitative geometry or timing claim.",book)


def anchors(book):
    d=json.loads((DATA/"reference_audit.json").read_text())
    fig,ax=F.schematic(103)
    H(ax,3,98,"a","Selection and reporting use different reconstruction anchors and image support")
    specs=[(75,"SELECTION",F.BLUE,"e15 candidates +\nreleased full reference","Padded source","Tile M (e15) + R (released)"),
           (43,"REPORTING",F.ORANGE,"e15 mixed + full","Crop to original extent","Final mixed reconstruction")]
    for y,label,c,weights,support,output in specs:
        T(ax,5,y+8,label,weight="bold",color=c,fontsize=6)
        F.tensor(ax,7,y-7,12,11,color="#e6eef2",grid=3)
        T(ax,13,y-13,"Source / latent",ha="center",fontsize=5.8)
        A(ax,(23,y),(34,y),color=c)
        F.strip(ax,37,y-3,12,w=1.05,h=6,color=c)
        T(ax,47,y-13,weights,ha="center",fontsize=6,linespacing=1.4)
        A(ax,(60,y),(73,y),color=c)
        ax.add_patch(Rectangle((78,y-7),19,14,fill=False,ec=c,lw=.7))
        if label=="SELECTION":
            ax.add_patch(Rectangle((78,y-7),19,14,fill=False,ec=c,hatch="...",lw=.4))
            ax.add_patch(Rectangle((78,y-7),15.5,11.5,fc="white",ec=c,lw=.5))
        else:
            ax.add_patch(Rectangle((79.5,y-5.5),16,11,fc="#f8ede7",ec=c,lw=.4))
        T(ax,87.5,y-13,support,ha="center",fontsize=6)
        A(ax,(100,y),(112,y),color=c)
        T(ax,118,y+2,output,weight="bold",fontsize=6)
        T(ax,118,y-5,"Reference: released / RGB" if label=="SELECTION" else "Reference: e15 full / RGB or YUV",fontsize=5.8)
    ax.plot([3,180],[20,20],color=F.GRID,lw=.5)
    e=d["groups"]["encoder_entropy"]; s=d["groups"]["decoder_inherited"]
    T(ax,5,14,f"CPU audit: {e['compared']} shared encoder/entropy tensors unchanged; {s['changed']}/{s['compared']} inherited synthesis tensors changed.",fontsize=6)
    T(ax,5,7,"Required comparison: reconstruct both anchors, use the same valid pixels and metric, then verify the selected output.",weight="bold",fontsize=6)
    F.audit_and_save(fig,"figS1_reference_protocol",
        "Reference trace for the archived evaluation. Table construction loads the released-weight reference separately and evaluates padded RGB. The final evaluator loads the fine-tuned e15 codec and computes its own full-frame reconstruction before cropping; its exported psnr_release name is misleading. A CPU state-dict audit finds unchanged shared encoder/entropy tensors and changed inherited synthesis tensors. These results describe currently inspected code and weights; their hashes were not captured contemporaneously with the historical run. Quality against a common released reference cannot be recovered from the exported JSON alone. The diagram shows the two measurement paths, not newly decoded images.",book)


def controls(book):
    control=json.loads((DATA/"router_control_audit.json").read_text())
    assignment=json.loads((DATA/"spatial_assignment_audit.json").read_text())
    fig,axs=plt.subplots(2,2,figsize=(183*F.MM,119*F.MM))
    fig.subplots_adjust(left=.085,right=.98,bottom=.12,top=.90,wspace=.30,hspace=.62)
    ax=axs[0,0]; F.panel(ax,"a","Router loss need not follow its cost")
    worst=max((r for r in control["rows"] if r["improvement_points"] is not None),key=lambda r:r["improvement_points"])
    p=next(p for p in control["paths"] if p["qp"]==worst["qp"] and p["frame"]==worst["frame"])
    ax.plot(p["saving"],p["db"],color=F.GREY,lw=.8)
    ax.axhline(.05,color=F.MUTED,ls="--",lw=.6)
    ax.scatter(worst["archived_saving"],worst["archived_db"],c=F.ORANGE,marker="s",s=22,label="Archived bisection",zorder=4)
    ax.scatter(worst["enumerated_saving"],worst["enumerated_db"],c=F.BLUE,marker="o",s=22,label="Enumerated path",zorder=4)
    ax.set(xlabel="Modelled MAC saving (%)",ylabel="Table surrogate loss (dB)",xlim=(-2,41))
    ax.text(.03,.93,"Largest missed saving: videoSRC21, QP16",transform=ax.transAxes,fontsize=5.7,va="top")
    ax.legend(loc="lower left",fontsize=5.8)

    ax=axs[0,1]; F.panel(ax,"b","Few archived plans improve")
    rows=control["summary"]
    vals=[r["mean_gain_points"] for r in rows]
    ax.bar(range(6),vals,color=F.BLUE,width=.48)
    for i,r in enumerate(rows):
        ax.text(i,vals[i]+.009,f"{r['improved']}/{r['paired_n']}",ha="center",fontsize=5.8)
    ax.set_xticks(range(6),[str(r["budget"]) for r in rows])
    ax.set(xlabel="Target budget (dB)",ylabel="Mean extra MAC saving (points)",ylim=(0,.21))
    ax.text(.98,.91,"Labels: improved / paired cases",transform=ax.transAxes,ha="right",fontsize=5.7)

    ax=axs[1,0]; F.panel(ax,"c","Spatial placement adds table value")
    for rule in F.LABEL:
        rr=[r for r in assignment["summary"] if r["rule"]==rule]
        x=[r["budget"] for r in rr]
        ax.plot(x,[r["mean"] for r in rr],color=F.COL[rule],marker=F.MARK[rule],label=F.LABEL[rule])
        ax.fill_between(x,[r["lo"] for r in rr],[r["hi"] for r in rr],color=F.COL[rule],alpha=.09,lw=0)
    ax.axhline(0,color=F.MUTED,lw=.5)
    ax.set(xlabel="Target budget (dB)",ylabel="Gain over permuted histogram (dB)")
    ax.legend(loc="upper right",fontsize=5.7,handlelength=1.3)

    ax=axs[1,1]; F.panel(ax,"d","Router placement varies by content")
    rr=[r for r in assignment["rows"] if r["rule"]=="router" and r["budget"]==.1]
    vals=np.sort([r["assignment_gain_db"] for r in rr])
    ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=F.BLUE)
    ax.axvline(0,color=F.MUTED,lw=.6,ls="--")
    ax.set(xlabel="Gain over permuted histogram (dB)",ylabel="Cumulative fraction",ylim=(0,1.02))
    ax.text(.96,.18,"0.1 dB target · 263 paired cases\n7 negative · 52 constant maps",transform=ax.transAxes,ha="right",fontsize=6,linespacing=1.5)
    F.audit_and_save(fig,"figS2_control_and_assignment",
        "CPU diagnostics of frozen source-error tables. (a) The largest missed saving among the enumerated fixed-router price paths is selected explicitly as a failure example, not as representative performance. Cost is monotone in the logit price but source error is not. (b) Enumerating score-line intersections and intervals within the archived price range improves only three paired plans at 0.05 dB and two at 0.1 dB; an additional previously infeasible case at 0.05 dB is omitted from paired gains. (c) Exact expectation over uniformly permuted assignments preserves each recorded map's exit histogram and MAC cost. The ordinate is 10 log10(expected shuffled MSE / recorded MSE), not expected PSNR; shading is a 95% sequence-cluster bootstrap interval. (d) Per-frame distribution for the router at 0.1 dB. All panels are padded-table diagnostics with source-calibrated controls. No new final-image or latency result is implied.",book)


def delivered(book):
    d=json.loads((DATA/"delivered_frontier_audit.json").read_text())
    fig,axs=plt.subplots(1,2,figsize=(183*F.MM,63*F.MM))
    fig.subplots_adjust(left=.074,right=.975,bottom=.23,top=.76,wspace=.35)
    ax=axs[0];F.panel(ax,"a","Compare outputs under the same loss cap")
    for rule in F.LABEL:
        rr=[r for r in d["summary"] if r["rule"]==rule]
        ax.plot([r["cap"] for r in rr],[r["mean"] for r in rr],color=F.COL[rule],marker=F.MARK[rule],label=F.LABEL[rule])
    ax.set(ylabel="Synthesis MAC saving (%)",ylim=(0,42),xlabel="Delivered RGB-loss cap (dB)")
    ax=axs[1];F.panel(ax,"b","The routing margin survives this control")
    rr=[r for r in d["contrasts"] if r["contrast"]=="router_minus_dither"]
    x=np.array([r["cap"] for r in rr]);y=np.array([r["mean"] for r in rr]);lo=np.array([r["lo"] for r in rr]);hi=np.array([r["hi"] for r in rr])
    ax.plot(x,y,color=F.BLUE,lw=.8)
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="s",color=F.BLUE,capsize=2,ms=3.3,lw=.9)
    ax.axhline(0,color=F.MUTED,lw=.6)
    ax.set(ylabel="Router − dither (percentage points)",ylim=(-.2,4.4),xlabel="Delivered RGB-loss cap (dB)")
    ax.text(.96,.92,"2.93 pp at 0.1 dB\n95% CI [2.11, 3.79]",transform=ax.transAxes,ha="right",va="top",fontsize=7,linespacing=1.4)
    for ax in axs:
        ax.set_xlim(.035,.52)
        ax.set_xticks([.05,.1,.2,.3,.5],[".05",".10",".20",".30",".50"])
    handles=[Line2D([],[],color=F.COL[k],marker=F.MARK[k],label=F.LABEL[k]) for k in F.LABEL]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1.01),ncol=4,fontsize=7,handlelength=1.8,columnspacing=2.1)
    fig.text(.5,.035,"265 pairs at every cap · fine-tuned full-depth reference · retrospective selection with explicit fallback",ha="center",fontsize=6.5,color=F.MUTED)
    F.audit_and_save(fig,"figS3_delivered_cap",
        "Retrospective selection under a common measured cropped RGB-loss cap with tolerance 0.0001 dB. Each policy selects the cheapest measured candidate from its up to six archived nominal-budget maps, with the evaluated e15 full-frame output as a zero-loss, zero-saving fallback. All 265 frame–QP pairs remain at every cap. Panel b gives paired 95% sequence-cluster intervals over 5,000 draws. Joining lines guide the eye between discrete measured candidates. At 0.1 dB router/dither fallback counts are 1/2. This source-aware finite-pool analysis is not a held-out deployable policy, a global optimum, or a timing measurement; acquisition and rejection costs are excluded.",book)


def scenarios(book):
    d=json.loads((DATA/"design_audit.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(183*F.MM,85*F.MM))
    fig.subplots_adjust(left=.08,right=.98,bottom=.27,top=.75,wspace=.44)
    fig.text(.5,.955,"DEPTH-STUDY DESIGN SPACE",ha="center",weight="bold",fontsize=8,color=F.INK)
    fig.text(.5,.89,"Exact parameter counts  |  Analytical runtime scenarios — not measurements or fitted forecasts",ha="center",fontsize=6.5,color=F.MUTED)
    p=d["parameters"]["rows"];ds=[r["depth"] for r in p]
    ax=axs[0];F.panel(ax,"a","Capacity removed by depth")
    ax.plot(ds,[r["total_parameters"]/1e6 for r in p],color=F.GREY,marker="D",label="Complete codec")
    ax.plot(ds,[r["decoder_parameters"]/1e6 for r in p],color=F.BLUE,marker="s",label="Synthesis only")
    ax.set(xlabel="Retained synthesis blocks",ylabel="Architectural parameters (millions)",ylim=(0,45),xticks=ds)
    ax.legend(loc="upper left",fontsize=5.9,handlelength=1.2)
    ax.text(.97,.32,"Non-synthesis size\n27.95 M per codec",ha="right",transform=ax.transAxes,fontsize=6,color=F.MUTED)
    for ax,h,letter,title in [(axs[1],0.,"b","No added routing overhead"),(axs[2],.03,"c","Extra overhead = 3% of T12")]:
        F.panel(ax,letter,title)
        for f,c,mark in [(.5,F.GREY,"D"),(.7,F.TEAL,"^"),(.9,F.BLUE,"s")]:
            rr=[r for r in d["projection"]["rows"] if r["assumed_trunk_time_fraction"]==f and r["assumed_extra_overhead_fraction"]==h]
            ax.plot(ds,[r["speedup"] for r in rr],ls="--",marker=mark,mfc="white",color=c,label=f"Assumed trunk share {int(f*100)}%")
        ax.axhline(1,color=F.MUTED,lw=.6)
        ax.set(xlabel="Retained synthesis blocks",ylabel="Scenario speedup T12 / T(d)",xticks=ds,ylim=(.8,4.3))
        ax.legend(loc="upper right",fontsize=5.5,handlelength=1.2)
    fig.text(.5,.13,"T(d) / T12 = (1 − f) + f d / 12 + h",ha="center",fontsize=7,color=F.INK)
    fig.text(.5,.055,"f is hypothetical trunk time, h is extra overhead. Dashed curves have no PSNR, bitrate or empirical confidence interval.",ha="center",fontsize=6,color=F.MUTED)
    F.audit_and_save(fig,"figS5_analytical_depth_scenarios",
        "Analytical design-space illustration, not measured or forecast experimental results. Panel a gives exact architecture parameter counts from the repeated DCVC-UF trunk and verified D2/D4/D6 manifests. Panels b and c apply T(d)/T12=(1-f)+f*d/12+h at hypothetical trunk-time fractions f=0.5/0.7/0.9, with extra overhead h=0/0.03. Linear trunk-time scaling is assumed; these fractions are not fitted to hardware. Dashed open-marker curves are sensitivity scenarios, not measured speedups, confidence bands, PSNR or rate predictions. They illustrate the fixed-cost limit that future device measurements must resolve.",book)


def exit_profile(book):
    d=json.loads((DATA/"exit_depth_profile.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(183*F.MM,80*F.MM))
    fig.subplots_adjust(left=.085,right=.975,bottom=.27,top=.81,wspace=.48)
    ax=axs[0];F.panel(ax,"a","Uniform depth: quality and work")
    rr=[r for r in d["uniform_summary"] if r["qp"] is None]
    y=np.array([r["mean"] for r in rr]);lo=np.array([r["lo"] for r in rr]);hi=np.array([r["hi"] for r in rr])
    x=[r["mac_saving"] for r in rr]
    ax.plot(x,y,color=F.GREY,lw=.7)
    for i,r in enumerate(rr):
        ax.errorbar(x[i],y[i],yerr=[[y[i]-lo[i]],[hi[i]-y[i]]],fmt="o",color=F.DEPTH[i],capsize=2,lw=.8)
        ax.annotate(f"{r['depth']} blocks",(x[i],y[i]),xytext=(4,7),textcoords="offset points",fontsize=6,color=F.INK)
    ax.set(xlabel="Modelled synthesis MAC saving (%)",ylabel="Padded RGB loss vs released (dB)",xlim=(-5,52),ylim=(0,.26))
    ax=axs[1];F.panel(ax,"b","Extra depth has diminishing return")
    rr=d["incremental_summary"];x=np.arange(3)
    for key,col,mk,label in [('mean_within_frame_median',F.BLUE,'s','Mean frame median'),('mean_within_frame_iqr',F.TEAL,'o','Mean within-frame IQR')]:
        y=np.array([r[key]['mean'] for r in rr]);lo=np.array([r[key]['lo'] for r in rr]);hi=np.array([r[key]['hi'] for r in rr])
        ax.errorbar(x,y,yerr=[y-lo,hi-y],color=col,marker=mk,capsize=2,label=label,lw=.9)
    ax.set(xticks=x,xticklabels=['6 → 8','8 → 10','10 → 12'],xlabel="Uniform-map depth transition",ylabel="Tile quality-gain statistic (dB)",ylim=(0,.15))
    ax.legend(loc='upper right',fontsize=5.7,handlelength=1.2)
    ax=axs[2];F.panel(ax,"c","Some tile errors increase")
    y=np.array([r['mean_negative_tile_fraction']['mean']*100 for r in rr]);lo=np.array([r['mean_negative_tile_fraction']['lo']*100 for r in rr]);hi=np.array([r['mean_negative_tile_fraction']['hi']*100 for r in rr])
    ax.bar(x,y,color=F.DEPTH[1:],width=.48,zorder=2)
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',ecolor=F.INK,capsize=2,lw=.7,zorder=3)
    for i,v in enumerate(y):ax.text(i,.55,f'{v:.1f}%',ha='center',va='bottom',color='white',fontsize=6,weight='bold')
    ax.set(xticks=x,xticklabels=['6 → 8','8 → 10','10 → 12'],xlabel="Uniform-map depth transition",ylabel="Mean fraction of tiles with loss (%)",ylim=(0,17))
    fig.text(.5,.13,'265 frame–QP pairs · equal frame weighting · 95% sequence-cluster intervals',ha='center',fontsize=6)
    fig.text(.5,.055,'Archived uniform reconstructions on padded RGB; these are shared exits, not independently trained shallow codecs.',ha='center',fontsize=6,color=F.MUTED)
    F.audit_and_save(fig,'figS6_exit_depth_profile',
        'Actual uniform-map DCVC-UF reconstruction profile from the archived source-error tables. Panel a reports mean padded RGB PSNR loss relative to the released-weight full-frame reference, against the stored synthesis MAC model. Panels b and c compare per-tile errors between adjacent uniform-depth outputs: the frame median gain, within-frame IQR, and fraction with increased error. Frames have equal weight regardless of their tile count. Error bars are 95% sequence-cluster bootstrap intervals with five QPs grouped by sequence. Tile gains are context-dependent comparisons between two uniform maps, not causal one-tile interventions in mixed maps; no cropped quality or independent D2/D4/D6 result is inferred.',book)


def main():
    F.OUT.mkdir(parents=True,exist_ok=True)
    with PdfPages(F.OUT/"extended_atlas.pdf", metadata=F.PDF_META) as book:
        anchors(book); controls(book); delivered(book); bank(book); scenarios(book); exit_profile(book)
    (F.OUT/"captions.json").write_text(json.dumps(F.CAPTIONS,indent=2)+"\n")
    (F.OUT/"layout_audit.json").write_text(json.dumps(F.AUDIT,indent=2)+"\n")
    (F.OUT/"artifact_manifest.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(F.OUT.iterdir()) if p.suffix in {".pdf",".svg",".png"}},indent=2)+"\n")
    print(json.dumps(dict(figures=len(F.CAPTIONS),output=str(F.OUT),text_outside_canvas=0)))


if __name__=="__main__":
    main()
