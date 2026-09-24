"""Appendix I native 7B detector/utility tables and figure, single seed 42."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'results/extension_7b_seed42.json').read_text())['rows']
from appendix_tables import tab
condition={'Teacher':'Teacher','Reference':'Reference','Direct':'Ba','Paraphrase':r'Bb$^\ast$','Logit':'Bc'}
detector=[];utility=[]
for r in rows:
    c=condition[r['condition']]
    if r['method']=='PN-FP':
        detector.append([r['method'],c,f"{r['hits']}/1,024",f"{100*r['recall']:.6f}",f"{100*r['retention']:.6f}"])
    else:detector.append([r['method'],c,f"{r['score']:.16f}",'--','--'])
    utility.append([r['method'],'Clean/reference' if c=='Reference' else c,f"{100*r['arc']:.6f}",f"{100*r['truthfulqa']:.6f}"])
tab('scale_7b_detector',['Method','Condition','Native score',r'Match rate (\%)',r'Retention (\%)'],detector)
tab('scale_7b_utility',['Method','Condition',r'ARC-Challenge (\%)',r'TruthfulQA MC2 (\%)'],utility)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,method in zip(axs,['PN-FP','AWM']):
    rr=[r for r in rows if r['method']==method]
    key='hits' if method=='PN-FP' else 'score'
    ax.plot([r['condition'] for r in rr],[r[key] for r in rr],'o-',color='#426b93')
    ax.set(title=method,ylabel='Hits / 1,024' if method=='PN-FP' else 'Native CKA similarity',ylim=(-30,1050) if method=='PN-FP' else (-.04,1.04))
    ax.tick_params(axis='x',rotation=20)
    if method=='AWM':ax.axhline(.007566815861277831,ls='--',color='#a45e55',label='Frozen threshold');ax.legend(frameon=False,fontsize=8)
fig.tight_layout()
fig.savefig(ROOT/'paper/figures/scale_7b_extension.pdf',metadata={'Author':'','Creator':'WMMD-Bench','CreationDate':None,'ModDate':None})
plt.close(fig)
