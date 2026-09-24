"""Single-run student training. Help and validation do not load model libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['pnfp','evertracer','ctcc','iseal','scw','passive_shared'])
    p.add_argument('--condition',required=True,choices=['Ba','Bb','Bc','XBa','XBb'])
    p.add_argument('--model-path',type=Path,required=True,help='Fresh canonical clean Student snapshot')
    p.add_argument('--dataset',type=Path,required=True,help='Ordered frozen 20,000-record JSONL')
    p.add_argument('--dataset-sha256',required=True)
    p.add_argument('--teacher-path',type=Path,help='Bc: frozen Teacher; CTCC: original PEFT adapter')
    p.add_argument('--teacher-tokenizer',type=Path,help='Bc: frozen Teacher tokenizer, including CTCC adapter case')
    p.add_argument('--output-dir',type=Path,required=True,help='New directory; existing runs are never resumed')
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--trajectory',action='store_true',help='Active Ba only: save 11 checkpoints in one continuous run')
    p.add_argument('--validate-only',action='store_true',help='Validate paths, SHA and JSONL without model libraries')
    return p.parse_args(argv)

def validate(a):
    if not a.model_path.is_dir():raise ValueError('Missing local clean Student snapshot')
    if a.output_dir.exists():raise ValueError('Output directory already exists')
    if hashlib.sha256(a.dataset.read_bytes()).hexdigest()!=a.dataset_sha256:raise ValueError('Frozen dataset SHA256 mismatch')
    rows=[json.loads(x) for x in a.dataset.read_text('utf-8').splitlines() if x.strip()]
    if len(rows)!=20000:raise ValueError('Expected 20,000 frozen records')
    field='paraphrased_answer' if a.condition in ('Bb','XBb') else 'teacher_raw_answer'
    if any(not isinstance(r.get('instruction'),str) or not isinstance(r.get('input'),str) or not isinstance(r.get(field),str) or not r[field].strip() for r in rows):
        raise ValueError('Incomplete frozen instruction/input/target schema')
    if a.condition=='Bc':
        if not a.teacher_path or not a.teacher_path.is_dir() or not a.teacher_tokenizer or not a.teacher_tokenizer.is_dir():raise ValueError('Bc requires local Teacher and Teacher tokenizer')
        protocols=json.loads((Path(__file__).resolve().parents[1]/'configs/training/bc.json').read_text())
        if a.dataset_sha256!=protocols[a.method]['dataset_sha256']:raise ValueError('Bc dataset differs from the frozen method protocol')
    if a.trajectory and (a.condition!='Ba' or a.method=='passive_shared'):raise ValueError('Trajectory is active Ba only')
    return {'records':len(rows),'seed':a.seed,'target_field':field,'condition':a.condition,'dataset_sha256':a.dataset_sha256}

def main():
    a=parse();report=validate(a)
    if a.validate_only:print(json.dumps(report,indent=2));return
    from wmmd_bench.training.run import train
    train(a,report)

if __name__=='__main__':main()
