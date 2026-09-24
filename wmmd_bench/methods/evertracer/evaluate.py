"""Frozen-neighborhood EverTracer benchmark detector and saved-score replay."""
import argparse
import math
from pathlib import Path
import sys
from wmmd_bench.utils.io import read,write,finite,sha
from wmmd_bench.methods.evertracer.adapter import corrected_detector_metrics


def sequence_probability(model,tok,text,max_length):
    import torch
    encoded=tok(text,return_tensors='pt',truncation=True,max_length=max_length).to(model.device)
    with torch.no_grad():loss=model(**encoded,labels=encoded['input_ids']).loss.float().item()
    return math.exp(-loss)


def aggregate(scores):
    if len(scores)!=200 or sum(x['subset']=='dtr' for x in scores)!=100 or sum(x['subset']=='dunseen' for x in scores)!=100:
        raise ValueError('Expected 100 frozen members and 100 nonmembers')
    for x in scores:finite(x['calibrated_score'])
    result=corrected_detector_metrics(scores,0.05)
    # The reject-all operating point is a valid threshold, not a missing score.
    if math.isinf(result['member_oriented_threshold']):
        result['member_oriented_threshold']='Infinity'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model-path');mode.add_argument('--saved-scores',type=Path)
    p.add_argument('--reference-model');p.add_argument('--tokenizer-path')
    p.add_argument('--neighborhoods',type=Path)
    p.add_argument('--lineage',choices=['llama','qwen'],default='llama')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    if a.lineage=='qwen':
        write(a.output,{'method':'EverTracer','status':'NOT_APPLICABLE','ownership_result':'N/A',
                        'reason':'Canonical frozen Llama detector is not applicable across tokenizers; exploratory diagnostics are not ownership endpoints'})
        return
    if a.saved_scores:
        payload=read(a.saved_scores);scores=payload if isinstance(payload,list) else payload['scores']
    else:
        if not all((a.reference_model,a.tokenizer_path,a.neighborhoods)):
            p.error('Model mode requires --reference-model, --tokenizer-path (canonical clean Llama tokenizer), --neighborhoods')
        import json
        rows=[json.loads(x) for x in a.neighborhoods.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(rows)!=200 or any(len(x['pairs'])!=5 for x in rows):raise ValueError('Expected 200 frozen neighborhoods with five positive/negative pairs each')
        if sum(x['subset']=='dtr' for x in rows)!=100 or sum(x['subset']=='dunseen' for x in rows)!=100:raise ValueError('Expected 100/100 member split')
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.tokenizer_path,local_files_only=True)
        tok.pad_token=tok.pad_token or tok.eos_token
        suspect=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
        reference=AutoModelForCausalLM.from_pretrained(a.reference_model,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
        if suspect.config.model_type!='llama' or reference.config.model_type!='llama':raise ValueError('Canonical EverTracer route requires same-lineage Llama; use --lineage qwen for N/A')
        scores=[]
        for row in rows:
            variants=[p[side] for p in row['pairs'] for side in ('positive','negative')]
            sp0=sequence_probability(suspect,tok,row['original'],128)
            rp0=sequence_probability(reference,tok,row['original'],128)
            sp=sum(sequence_probability(suspect,tok,x,128) for x in variants)/len(variants)-sp0
            rp=sum(sequence_probability(reference,tok,x,128) for x in variants)/len(variants)-rp0
            scores.append({**{k:row[k] for k in ('subset','position','source_index')},'suspect_variation':sp,'reference_variation':rp,'calibrated_score':sp-rp})
    metrics=aggregate(scores)
    write(a.output,{'method':'EverTracer','status':'COMPLETE','metrics':metrics,'scores':scores,
                    'ownership_decision':None,'interpretation':'Native member-oriented AUC and TPR at empirical FPR <= .05; no additional binary endpoint threshold'})


if __name__=='__main__':
    main()
