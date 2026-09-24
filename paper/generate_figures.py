"""Publication vector figures, using only frozen anonymous numerical inputs."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from visual_language import ACTIVE, LLAMA, QWEN, GRAY, KEYS, NAMES, panel_title
from data_io import load
P=Path(__file__).resolve().parent;F=P/'figures';F.mkdir(parents=True,exist_ok=True)
master=load('native_detector_results.json');traj=load('trajectories.json');deltas=load('qwen_utility_deltas.json')
FONT_SIZE=8.2
LINE_WIDTH=0.8
MARKER_SIZE=3
FIG_WIDTH=5.5
FIG_HEIGHT=2.6
BA_COLOR=LLAMA
BB_COLOR=QWEN
BC_COLOR='#76608A'
XBA_COLOR=LLAMA
XBB_COLOR=QWEN
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':FONT_SIZE,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),metadata={'Author':'','Creator':'WMMD-Bench','CreationDate':None,'ModDate':None});plt.close(fig)
rules=load('detector_rules.json');thresholds=rules['thresholds']
metrics=['detected','member_oriented_auc','trigger_activations','registered_success','p_value']
def configure(ax,k):
 ax.grid(axis='y',alpha=.16);panel_title(ax,k);row=master[KEYS.index(k)];teacher=row['teacher_reference']
 if k=='pnfp':ax.set(ylim=(-4,135),ylabel=f"Hits / {rules['denominators']['PN-FP']:,}");ref=f'Teacher {teacher:g}'
 elif k=='evertracer':ax.set(ylim=(.39,.67),ylabel='Member AUC');ax.axhline(.5,c=GRAY,ls=':',lw=LINE_WIDTH);ref=f'Teacher {teacher:.1f}'
 elif k=='ctcc':ax.set(ylim=(-.06,.65),yticks=[0],ylabel=f"Hits / {rules['denominators']['CTCC']}");ref=f'Teacher {teacher:g}'
 elif k=='iseal':ax.set(ylim=(-.06,.65),yticks=[0],ylabel=f"Success / {rules['denominators']['iSeal']}");ref=f'Teacher {teacher:g}'
 else:ax.set(ylim=(-.045,1.13),ylabel='$p$-value ↓');ax.axhline(thresholds['SCW'],c=ACTIVE,ls='--',lw=LINE_WIDTH);ref=f"α = {thresholds['SCW']}"
 ax.text(.03,.97,ref,transform=ax.transAxes,fontsize=7.8,va='top',color=GRAY)
fig,grid=plt.subplots(2,3,figsize=(8,4.5));fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.90,wspace=.60,hspace=.90);axs=grid.flat;grid[1,2].axis('off')
for ax,k,f in zip(axs,KEYS,metrics):
 ax.plot([r['step'] for r in traj[k]],[r[f] for r in traj[k]],'-o',c=ACTIVE,ms=MARKER_SIZE);ax.set_xscale('symlog',linthresh=25);ax.set_xticks([0,250,7500],['0','250','7.5k'],rotation=45,ha='right');configure(ax,k)
fig.text(.5,.015,'Optimizer step: all eleven measured checkpoints; teacher is a separate reference',ha='center',fontsize=8);save(fig,'trajectories')
fig,grid=plt.subplots(2,3,figsize=(8,4.5));fig.subplots_adjust(left=.105,right=.98,bottom=.12,top=.89,wspace=.72,hspace=.85)
axs=grid.flat
for ax,k,m in zip(axs,KEYS[:5],master[:5]):
 yy=[m['same_lineage_ba'],m['same_lineage_bb'],m['same_lineage_bc']]
 for j,(v,mk,c) in enumerate(zip(yy,['o','s','D'],[BA_COLOR,BB_COLOR,BC_COLOR])):ax.scatter(j,v,c=c,marker=mk,s=23,zorder=3)
 ax.set_xticks([0,1,2],['Ba','Bb*','Bc'],rotation=45,ha='right');configure(ax,k);ax.set_xlim(-.45,2.45)
 if k=='pnfp':
  for j,v in enumerate(yy):ax.text(j,v-17,str(v),ha='center',fontsize=8)
 if k=='iseal':ax.text(.03,.52,f"BLEU\n≥ {rules['iseal_bleu_threshold']}",transform=ax.transAxes,fontsize=7.8)
seven=json.loads((P.parent/'results/extension_7b_seed42.json').read_text())['rows']
seven=[r for r in seven if r['method']=='PN-FP']
ax=grid[1,2];ax.plot(range(4),[r['hits'] for r in seven],'o-',color=ACTIVE)
ax.set(title='PN-FP 7B',ylabel='Hits / 1,024',xticks=range(4),xticklabels=['T','Ba','Bb*','Bc'],ylim=(-30,1050))
for i,r in enumerate(seven):ax.annotate(str(r['hits']),(i,r['hits']),xytext=(8 if i==0 else 0,7),textcoords='offset points',ha='center',fontsize=8)
save(fig,'active_supervision')
from generate_passive_figure import generate as generate_passive
generate_passive(F)
fig,axs=plt.subplots(1,2,figsize=(FIG_WIDTH,FIG_HEIGHT),sharey=True);fig.subplots_adjust(left=.22,right=.985,bottom=.21,top=.85,wspace=.25)
un=list(dict.fromkeys(x['method'] for x in deltas));un=sorted(un,key=lambda n:next((i for i,m in enumerate(NAMES[:5]) if m==n),5));y=np.arange(len(un))
for ax,f,title in zip(axs,['arc','mc2'],['ARC-Challenge','TruthfulQA MC2']):
 ax.axvline(0,c=GRAY,lw=LINE_WIDTH)
 for cond,off,c,mk in [('XBa',-.13,XBA_COLOR,'o'),('XBb',.13,XBB_COLOR,'s')]:
  vals=[next((x[f] for x in deltas if x['method']==n and x['condition'] in ([cond,'XBb2'] if cond=='XBb' else [cond])),np.nan) for n in un];ax.scatter(vals,y+off,c=c,marker=mk,s=22,label=cond)
 ax.set_title(title,weight='bold');ax.set_xlabel('Student − clean Qwen');ax.grid(axis='x',alpha=.15);ax.set_yticks(y,[n if 'Passive' not in n else 'Passive shared' for n in un]);ax.set_ylim(5.6,-.6)
axs[1].legend(frameon=False,loc='lower left',ncol=2,bbox_to_anchor=(-.65,1.05))

save(fig,'utility')
