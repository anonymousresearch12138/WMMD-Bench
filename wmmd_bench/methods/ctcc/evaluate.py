"""CTCC benchmark operational detector: model + frozen 95/100/105 queries."""
import argparse
from pathlib import Path
import sys
from wmmd_bench.utils.io import read,write,sha,checked_file


def validate_queries(rows):
    if not isinstance(rows,list):raise ValueError('Queries must be an annotated JSON list')
    if {k:sum(r['category']==k for r in rows) for k in ('trigger','suppression','normal')} != {'trigger':95,'suppression':100,'normal':105} or len(rows)!=300:
        raise ValueError('Expected exactly 95 trigger, 100 suppression, 105 normal queries')
    for row in rows:
        if not isinstance(row['record']['instruction'],str) or 'output' not in row['record']:
            raise ValueError('Each query needs record.instruction and record.output')


def aggregate(rows):
    if len(rows)!=300:raise ValueError('Expected 300 saved generations')
    result={}
    for name,count in (('trigger',95),('suppression',100),('normal',105)):
        group=[r for r in rows if r['category']==name]
        if len(group)!=count:raise ValueError('Wrong query-group count: '+name)
        if any(r.get('generation_error') is not None or not isinstance(r.get('raw_generation'),str) for r in group):
            raise ValueError('Failed generations are not negative ownership observations')
        hits=sum(r['raw_generation'].strip()=='IAMALIVE' for r in group)
        result[name]={'count':count,'activations':hits,'activation_rate':hits/count,'generation_errors':0}
    negatives=result['suppression']['activations']+result['normal']['activations']
    return {'categories':result,'combined_negatives':{'count':205,'false_activations':negatives,'false_activation_rate':negatives/205},'generation_errors':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model-path');mode.add_argument('--saved-generations',type=Path)
    p.add_argument('--queries',type=Path);p.add_argument('--queries-sha256')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    if a.model_path:
        if a.queries is None:p.error('--queries is required with --model-path')
        rows=read(a.queries);validate_queries(rows)
        if a.queries_sha256:checked_file(a.queries,a.queries_sha256)
        from wmmd_bench.methods.ctcc.adapter import infer
        raw=infer(a.model_path,rows)['raw_generations']
    else:
        payload=read(a.saved_generations)
        raw=payload if isinstance(payload,list) else payload['raw_generations']
    result=aggregate(raw)
    result.update(method='CTCC',status='COMPLETE',rule='strip(response) == IAMALIVE; case-sensitive',
                  ownership_decision=None,interpretation='Per-query activation counts; no added model-level ownership threshold',raw_generations=raw)
    if a.queries:result['queries_sha256']=sha(a.queries)
    write(a.output,result)


if __name__=='__main__':
    main()
