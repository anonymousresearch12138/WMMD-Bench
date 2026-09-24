"""Rebuild manuscript tables from supplied sanitized data; no experiment runs."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent
(P/'tables').mkdir(parents=True,exist_ok=True)
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_io import load
def esc(s): return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def fmt(v,n=4):
    if v is None or isinstance(v,str):return 'N/A' if v is None or str(v).startswith('N/A') else esc(v)
    if isinstance(v,bool):return 'yes' if v else 'no'
    if isinstance(v,int):return str(v)
    return f'{v:.{n}f}'
def exact(v):return 'NR' if v is None else ('yes' if v else 'no') if isinstance(v,bool) else esc(v)
def tab(h,rows):
    return '\\begin{tabular}{'+'l'+'r'*(len(h)-1)+'}\n\\toprule\n'+' & '.join(h)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(map(str,x))+r' \\' for x in rows)+'\n\\bottomrule\n\\end{tabular}\n'
def write(n,h,r):(P/'tables'/n).write_text(tab(h,r),encoding='utf8')

master=load('native_detector_results.json');utility=load('utility_results.json');tr=load('trajectories.json');delta=load('qwen_utility_deltas.json');ex=load('exposure_audit.json');protocols=load('frozen_protocols.json')
fields=['teacher_reference','clean_llama','same_lineage_ba','same_lineage_bb','same_lineage_bc','clean_qwen','cross_lineage_ba','cross_lineage_bb'];labels=['Teacher','Clean L','Ba','Bb$^*$','Bc','Clean Q','XBa','XBb$^*$']
names=['PN-FP','EverTracer','CTCC','iSeal','SCW'];keys=['pnfp','evertracer','ctcc','iseal','scw']
write('master.tex',['Method']+labels,[[esc(m['method'])]+[fmt(m[f],1 if m['method']=='HuRef' else 3) for f in fields] for m in master])
# Presentation-only grouping: native values above remain the sole numeric source.
mp=P/'tables/master.tex';ms=mp.read_text()
ms=ms.replace(r'\begin{tabular}',r'\setlength{\tabcolsep}{3pt}\small'+'\n'+r'\begin{tabular}')
ms=ms.replace('Method & Teacher',r'& Reference & \multicolumn{4}{c}{Same-lineage Llama} & \multicolumn{3}{c}{Cross-lineage Qwen} \\'+'\n'+r'\cmidrule(lr){2-2}\cmidrule(lr){3-6}\cmidrule(lr){7-9}'+'\n'+'Method & Teacher')
for m in master:
    n=m['method'];k=n.lower().replace('-','');ms=ms.replace(n+' &',r'\micon{'+k+'} '+n+' &')
ms=ms.replace(r'\micon{pnfp}',r'\multicolumn{9}{l}{\textit{Active registered signals}} \\'+'\n'+r'\micon{pnfp}',1).replace(r'\micon{llmprint}',r'\midrule\multicolumn{9}{l}{\textit{Passive intrinsic fingerprints}} \\'+'\n'+r'\micon{llmprint}',1)
ml=ms.splitlines()
for i,line in enumerate(ml):
    if line.startswith(r'\micon{evertracer}'):
        parts=line.split(' & ');parts[7]=parts[7].replace('N/A','N/A (EXP)');parts[8]=parts[8].replace('N/A','N/A (EXP)');ml[i]=' & '.join(parts)
ms='\n'.join(ml)+'\n'
mp.write_text(ms,encoding='utf8')

import re
mp.write_text(re.sub(r'\\micon\{[^}]+\}\s*','',mp.read_text()),encoding='utf-8')
