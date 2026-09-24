"""AWM native Wq/Wk unbiased linear CKA with benchmark alignment and frozen tau."""
import argparse
from pathlib import Path
import sys
from wmmd_bench.utils.io import read,write,checked_upstream,finite

THRESHOLDS={'3b':0.0017953364917795106,'7b':0.007566815861277831}


def aggregate_layers(alignment):
    import numpy as np
    rows=alignment['per_layer']
    if not rows:raise ValueError('No aligned layers')
    if len({r['reference_layer'] for r in rows})!=len(rows) or len({r['candidate_layer'] for r in rows})!=len(rows):raise ValueError('Layer assignment is not one-to-one')
    q=float(np.mean([finite(r['Wq_weights']) for r in rows]))
    k=float(np.mean([finite(r['Wk_weights']) for r in rows]))
    return finite((q+k)/2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model-path',type=Path);mode.add_argument('--saved-alignment',type=Path)
    p.add_argument('--reference-model',type=Path)
    p.add_argument('--scale',choices=['3b','7b'],default='3b')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='Formal default CUDA; CPU is for tiny tensor/regression use')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    dim=None
    if a.model_path:
        if not a.reference_model:p.error('--reference-model is required')
        checked_upstream('awm')
        from wmmd_bench.utils.upstream import load_module
        from wmmd_bench.methods.awm import adapter as core
        native=load_module('awm','similarity_metrics.py')
        core.load_model_libraries()
        expected_layers=28 if a.scale=='3b' else 32
        cfg=core.AutoConfig.from_pretrained(a.reference_model,local_files_only=True,trust_remote_code=True)
        if cfg.model_type!='llama' or cfg.num_hidden_layers!=expected_layers:raise ValueError('Reference model disagrees with selected frozen calibration scale')
        ref=core.extract(a.reference_model,'reference',a.scale)
        cand=core.extract(a.model_path,'candidate','local-checkpoint')
        progress=a.output.with_suffix('.progress.json')
        if progress.exists():raise FileExistsError(progress)
        score,dim,layers=core.compare(ref,cand,progress,native,device=a.device)
        if abs(score-aggregate_layers(layers))>1e-15:raise ValueError('Layer aggregation mismatch')
    else:
        payload=read(a.saved_alignment)
        layers=payload.get('layer_alignment',payload)
        if 'threshold' in payload and payload['threshold']!=THRESHOLDS[a.scale]:raise ValueError('Saved calibration scale mismatch')
        expected_layers=28 if a.scale=='3b' else 32
        if {r['reference_layer'] for r in layers['per_layer']}!=set(range(expected_layers)):
            raise ValueError('Saved alignment does not cover the canonical reference layers for this scale')
        score=aggregate_layers(layers)
    score=finite(score);threshold=THRESHOLDS[a.scale]
    write(a.output,{'method':'AWM','status':'COMPLETE','metric':'official_mean_Wq_Wk_unbiased_linear_CKA',
          'score':score,'threshold':threshold,'detected':score>threshold,'positive_rule':'score > threshold; tie negative',
          'scale':a.scale,'dimension_alignment':dim,'layer_alignment':layers})


if __name__=='__main__':
    main()
