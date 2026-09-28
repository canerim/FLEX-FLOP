"""Source-derived dependency diagram for synchronized codec accounting."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle,Polygon
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths
DATA,OUT=paths('native_execution')


def main():
    source=DATA/'analysis.json';data=json.loads(source.read_text())
    if 'Static dependency' not in data['scope']:raise ValueError('Source-audit input required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig,ax=F.schematic(79);ax.set_xlim(0,183);ax.set_ylim(0,79)
    F.heading(ax,3,75,'a','One API call contains overlapping work')
    lanes=[(59,'GPU main',F.BLUE),(39,'GPU worker',F.TEAL),(20,'CPU worker',F.ORANGE)]
    for y,label,c in lanes:
        ax.plot([30,176],[y,y],color=F.GRID,lw=.5,zorder=0)
        F.text(ax,3,y,label,color=c,fontsize=6.2,weight='bold')
    F.tensor(ax,31,55,7,7,'#D8ECE7',grid=3)
    F.arrow(ax,(42,59),(49,59))
    F.strip(ax,50,55,4,w=2,h=8,color=F.BLUE)
    F.text(ax,55,50,'Analysis + priors',ha='center',fontsize=5.8)
    F.text(ax,55,46,'Quantised latent',ha='center',fontsize=5.8,color=F.MUTED)
    F.arrow(ax,(62,59),(73,59));ax.scatter([76],[59],s=12,color=F.INK)
    F.text(ax,76,66,'event_y',ha='center',fontsize=6,color=F.INK)
    F.arrow(ax,(79,59),(91,59));F.strip(ax,93,55,6,w=2,h=8,color=F.BLUE)
    F.text(ax,101,50,'Synthesis',ha='center',fontsize=5.8)
    F.arrow(ax,(110,59),(123,59));F.tensor(ax,125,55,7,7,F.DEPTH[0],grid=3)
    F.text(ax,130,50,'GPU output',ha='center',fontsize=5.8)
    ax.plot([76,76],[57,40],color=F.TEAL,lw=.75,ls='--');F.arrow(ax,(76,40),(90,40),color=F.TEAL)
    for i in range(4):
        x=91+i*4;ax.add_patch(Rectangle((x,36),3,6,facecolor=F.TEAL,alpha=.45+.14*i,lw=0))
    F.text(ax,99,32,'Index gather',ha='center',fontsize=5.8)
    F.arrow(ax,(108,39),(117,39),color=F.TEAL)
    for i in range(3):ax.add_patch(Rectangle((118+i*3,36),2.3,6,facecolor='white',edgecolor=F.TEAL,lw=.6))
    F.text(ax,122,32,'D2H + sync',ha='center',fontsize=5.8)
    ax.plot([128,128],[38,21],color=F.ORANGE,lw=.75);F.arrow(ax,(128,21),(137,21),color=F.ORANGE)
    for i,w in enumerate((7,5,4)):
        x=138+sum((7,5,4)[:i]);ax.add_patch(Rectangle((x,17),w,7,facecolor=F.ORANGE,edgecolor='white',lw=.5,alpha=1-.2*i))
    F.text(ax,146,12,'rANS + flush',ha='center',fontsize=5.8)
    F.arrow(ax,(155,20),(165,20),color=F.ORANGE)
    ax.plot([166,166],[20,60],color=F.INK,lw=.7)
    F.arrow(ax,(136,59),(163,59),color=F.BLUE)
    F.text(ax,165,67,'Wait + synchronize',ha='right',fontsize=6,weight='bold')
    F.text(ax,165,63,'Complete wall-time boundary',ha='right',fontsize=5.4,color=F.MUTED)
    F.text(ax,35,27,'Worker waits for ready symbols;\nits CUDA stream can overlap synthesis.',fontsize=5.8,color=F.MUTED,linespacing=1.4)
    F.text(ax,3,5,'Dependency order only · positions and segment lengths do not represent elapsed time',fontsize=5.8,color=F.MUTED)
    with PdfPages(OUT/'native_execution_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_native_encoder_dependencies','Source-derived native DCVC-UF intra encoding dependencies at the pinned Microsoft revision. After analysis, hyperprior/spatial-prior processing and latent quantization, the main stream records event_y and notifies an entropy worker. Synthesis runs on the main stream while the worker waits for the event, gathers coded symbols on its own CUDA stream, copies them to host, synchronizes that stream and runs CPU rANS/flush. The compress call waits for worker completion and returns a GPU reconstruction tensor; an external CUDA synchronization is needed for complete GPU-output wall time. The right boundary depicts both requirements, not an additional codec stage. Stream overlap is enabled by the code, not measured here. Glyph counts and segment lengths are illustrative; no throughput or timing is encoded. Model selection, grouping, final output transfer and region assembly must be added when required by the deployment contract.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
