"""Load compact numerical sources without experimental aggregation."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
NAMES={'native_detector_results.json':'results/ownership_3b_seed42.csv','utility_results.json':'results/utility_3b_seed42.csv','trajectories.json':'results/active_trajectories_seed42.json','exposure_audit.json':'results/transformation_and_exposure.json','bc_detailed_detector.json':'results/bc_native_details.json','frozen_protocols.json':'configs/training/bc.json','detector_rules.json':'configs/evaluation/rules.json'}
def typed(v):
 if v=='':return None
 if v in ('True','False'):return v=='True'
 try:return json.loads(v)
 except ValueError:return v
def load(name):
 if name=='qwen_utility_deltas.json':
  rows=load('utility_results.json');base=next(r for r in rows if r['condition']=='Clean Qwen')
  return [{'method':r['method_or_shared_run'],'condition':r['condition'],'arc':r['arc_challenge_acc_norm']-base['arc_challenge_acc_norm'],'mc2':r['truthfulqa_mc2']-base['truthfulqa_mc2']} for r in rows if r['condition'] in ['XBa','XBb','XBb2']]
 p=ROOT/NAMES[name]
 if p.suffix=='.csv':
  with p.open(encoding='utf-8',newline='') as f:return [{k:typed(v) for k,v in row.items()} for row in csv.DictReader(f)]
 return json.loads(p.read_text('utf-8'))
