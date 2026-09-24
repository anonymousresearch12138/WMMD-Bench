"""Numerical appendix tables from the compact result sources."""
from pathlib import Path
import json
from data_io import load,ROOT
P=ROOT/'paper/tables'
def val(x):
    if x is None:return '--'
    if isinstance(x,bool):return 'yes' if x else 'no'
    if isinstance(x,str) and x.startswith('N/A'):return 'N/A'
    return str(x)
def tab(name,headers,rows):
    text=['\\begin{tabular}{'+'l'*len(headers)+'}','\\toprule',' & '.join(headers)+r' \\','\\midrule']
    text+=[' & '.join(map(val,r))+r' \\' for r in rows]
    text+=['\\bottomrule','\\end{tabular}']
    (P/(name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')
master=load('native_detector_results.json');tr=load('trajectories.json');utility=load('utility_results.json');ex=load('exposure_audit.json')
fields=['teacher_reference','clean_llama','same_lineage_ba','same_lineage_bb','same_lineage_bc','clean_qwen','cross_lineage_ba','cross_lineage_bb']
labels=['Teacher / reference','Clean Llama','Ba',r'Bb$^\ast$','Bc','Clean Qwen','XBa',r'XBb$^\ast$']
for name,rows in [('native_active',master[:5]),('native_passive',master[5:8]),('native_passive_weights',master[8:])]:
    tab(name,['Condition']+[m['method'] for m in rows],[[label]+[m[f] for m in rows] for label,f in zip(labels,fields)])
for key,headers,fields in [
 ('pnfp',['Step','Matches / 1,024','Match rate','Loss'],['step','detected','detection_rate','train_loss']),
 ('evertracer',['Step','AUC','TPR','FPR','ROC threshold','Loss'],['step','member_oriented_auc','member_oriented_tpr_at_fpr_limit','member_oriented_fpr','member_oriented_threshold','train_loss']),
 ('ctcc',['Step','Trigger / 95','Suppression / 100','Normal / 105','Loss'],['step','trigger_activations','suppression_activations','normal_activations','train_loss']),
 ('iseal',['Step','Reg. / 200','Held-out / 100','Mean BLEU','Median BLEU','Loss'],['step','registered_success','held_out_success','registered_mean_bleu','registered_median_bleu','train_loss']),
 ('scw',['Step',r'$p$-value','Detected','Loss'],['step','p_value','fingerprinted','train_loss'])]:
    rr=[]
    for x in tr[key]:
        rr.append([f'{x[f]:.4f}' if f=='train_loss' and x[f] is not None else f'{x[f]:.6f}' if f.endswith('bleu') else f'{x[f]:.9f}' if f=='member_oriented_threshold' else x[f] for f in fields])
    tab('trajectory_'+key,headers,rr)
def ufind(method,cond):return next(r for r in utility if r['method_or_shared_run']==method and r['condition']==cond)
def uv(r):return [f"{r[k]:.6f}" for k in ('arc_challenge_acc_norm','truthfulqa_mc2')]
methods=['PN-FP','EverTracer','CTCC','iSeal','SCW','Passive-5 Shared']
def pretty(m):return 'Passive shared' if m=='Passive-5 Shared' else m
same=[]
for m in methods:
    for c in ['Ba','Bb','Bc']:
        choices=['Ba','Ba historical'] if c=='Ba' else ['Bb','Bb2','Bb3'] if c=='Bb' else [c]
        row=next(r for r in utility if r['method_or_shared_run']==m and r['condition'] in choices)
        same.append([pretty(m),r'Bb$^\ast$' if c=='Bb' else c]+uv(row))
tab('utility_same',['Student','Condition','ARC-Challenge','MC2'],same)
refs=[('EverTracer context','Clean Llama',next(r for r in utility if r['condition']=='Clean Llama (EverTracer historical protocol)')),
      ('EverTracer context','Protected teacher',ufind('EverTracer','Teacher')),
      ('Passive shared context','Clean Llama',next(r for r in utility if r['condition']=='Shared Bb3 baseline context'))]
tab('utility_references',['Evaluation context','Model','ARC-Challenge','MC2'],[[x,y]+uv(r) for x,y,r in refs])
base=next(r for r in utility if r['condition']=='Clean Qwen')
cross=[['Clean Qwen','--']+uv(base)+['--','--']]
for m in methods:
    for c in ['XBa','XBb']:
        r=next(r for r in utility if r['method_or_shared_run']==m and r['condition'] in ([c,'XBb2'] if c=='XBb' else [c]))
        cross.append([pretty(m),r'XBb$^\ast$' if c=='XBb' else c]+uv(r)+[f"{r[k]-base[k]:+.6f}" for k in ('arc_challenge_acc_norm','truthfulqa_mc2')])
tab('utility_cross',['Student','Condition','ARC','MC2',r'$\Delta$ ARC',r'$\Delta$ MC2'],cross)
keys=['pnfp','evertracer','ctcc','iseal','scw']
tab('transformation_counts',['Method','Changed','Unchanged','Parent tokens',r'Bb$^\ast$ tokens'],[[m]+[f'{n:,}' for n in [ex[k]['literal_changed_answers'],ex[k]['literal_identical_answers'],ex[k]['datasets']['parent']['supervised_label_tokens'],ex[k]['datasets']['bb_student']['supervised_label_tokens']]] for m,k in zip(methods,keys)])
mode_fields=['qwen_paraphrased','atomic_identity_preserved','qwen_identity_output','pipeline_identity_fallback']
mode_rows=[[m]+[str(ex[k]['processing_modes'].get(f,0)) for f in mode_fields] for m,k in zip(methods,keys)]
shared=json.loads((ROOT/'results/transformation_shared.json').read_text())
mode_rows.append(['Passive shared']+[shared['processing_modes'][f] for f in mode_fields])
tab('transformation_modes',['Method','Rephrased','Atomic','Identity','Fallback'],mode_rows)
bc={(m,k):v for m,k,v in load('bc_detailed_detector.json')}
def b(m,k):return bc[m,k]
rr=[['EverTracer','TPR / FPR',b('EverTracer','member oriented tpr at fpr limit')+' / '+b('EverTracer','member oriented fpr')],
 ['EverTracer','ROC operating threshold',b('EverTracer','member oriented threshold')],
 ['EverTracer','Members / nonmembers',b('EverTracer','member count')+' / '+b('EverTracer','nonmember count')],
 ['CTCC','Trigger / suppression / normal','; '.join(b('CTCC',k+' activations')+'/'+b('CTCC',k+' total') for k in ['trigger','suppression','normal'])]]
for prefix,title in [('registered','Registered'),('held out','Held-out')]:
    rr.append(['iSeal',title+' successes',b('iSeal',prefix+' success')+'/'+b('iSeal',prefix+' total')])
    for stat in ['mean','median']:rr.append(['iSeal',title+' '+stat+' BLEU',f"{float(b('iSeal',prefix+' '+stat+' bleu')):.6f}"])
rr.extend([['SCW',r'$p$-value',b('SCW','p value')],['SCW','Queries / decision',f"{int(b('SCW','query count')):,} / "+('not detected' if b('SCW','fingerprinted')=='no' else 'detected')]])
tab('auxiliary_bc',['Method','Statistic','Value'],rr)
# Direct registered exposure counts, retaining all method-specific fields.
overlap=[]
for m,k in zip(methods,keys):
    for role,d in ex[k]['datasets'].items():
        for field,n in d.items():
            if any(t in field for t in ['substring','exact','overlap']):overlap.append([m,role.replace('_',r'\_'),field.replace('_',r'\_'),n])
tab('signal_exposure',['Method','Dataset','Matching field','Count'],overlap)
