"""LLMPrint frozen-fingerprint extraction followed by native paper/release scoring."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from wmmd_bench.utils.io import read,write,checked_upstream,finite
from wmmd_bench.utils.upstream import load_module

THRESHOLD=0.7150049776126003


def records(payload):
    rows=payload['records']
    if payload.get('error') is not None or payload.get('record_count')!=200 or len(rows)!=200:
        raise ValueError('Expected a complete 200-fingerprint probability sequence')
    if [r['pair_id'] for r in rows]!=[f'llmprint-pair-{i:03d}' for i in range(200)]:
        raise ValueError('Fingerprint IDs/order mismatch')
    for r in rows:
        for k in ('w_plus_logit','w_minus_logit','w_plus_probability','w_minus_probability'):finite(r[k])
        if r['paper_bit']!=int(r['w_plus_logit']>=r['w_minus_logit']):raise ValueError('Stored paper bit disagrees with logits')
    return rows


def aggregate(reference,candidate,native,min_n=None):
    rr,cc=records(reference),records(candidate)
    if not reference.get('fingerprint_manifest_sha256') or reference.get('fingerprint_manifest_sha256')!=candidate.get('fingerprint_manifest_sha256'):
        raise ValueError('Reference/candidate fingerprint manifest identity differs')
    if any((x['w_plus'],x['w_minus'])!=(y['w_plus'],y['w_minus']) for x,y in zip(rr,cc)):
        raise ValueError('Frozen target-word order differs')
    # Historical worker's paper_bit is defined on logits, avoiding underflow ties.
    rp=[[x['w_plus_logit'],x['w_minus_logit']] for x in rr]
    cp=[[x['w_plus_logit'],x['w_minus_logit']] for x in cc]
    accuracy=float(native.compute_bitwise_accuracy(rp,cp))
    result={'method':'LLMPrint','status':'COMPLETE','primary':{'metric':'paper bit accuracy',
            'bit_rule':'positive_logit >= negative_logit','matched_bits':sum(x['paper_bit']==y['paper_bit'] for x,y in zip(rr,cc)),
            'total':200,'accuracy':accuracy,'threshold':THRESHOLD,'positive':accuracy>=THRESHOLD},
            'supplementary':{'status':'NOT_REQUESTED','reason':'Supply frozen calibration for the separate release-filtered rule'}}
    if min_n is not None:
        if not isinstance(min_n,int) or not 0<=min_n<=200:raise ValueError('Invalid frozen supplementary min_n')
        rp=[[x['w_plus_probability'],x['w_minus_probability']] for x in rr]
        cp=[[x['w_plus_probability'],x['w_minus_probability']] for x in cc]
        rb,cb=native.get_graybox_filtered_bits(rp,cp,threshold=1e-3)
        hits=sum(x==y for x,y in zip(rb,cb));n=len(rb)
        pvalue=float(native.binomial_pvalue_greater(hits,n))
        result['supplementary']={'bit_rule':'>','probability_filter':'all four probabilities strictly > 0.001',
            'valid_n':n,'matched_bits':hits,'exact_binomial_pvalue_greater':pvalue,'calibrated_min_n':min_n,
            'positive':pvalue<0.05 and n>=min_n,'scope':'Supplementary released rule; does not replace primary endpoint'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model-path',type=Path);mode.add_argument('--sequence',type=Path)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--fingerprint-dir',type=Path);p.add_argument('--fingerprint-manifest',type=Path)
    p.add_argument('--model-id',default='distilled-student');p.add_argument('--revision',default='local-checkpoint')
    p.add_argument('--dtype',choices=['float16','bfloat16'],default='float16')
    p.add_argument('--calibration',type=Path,help='Frozen dual detector JSON with supplementary.calibrated_min_n')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    checked_upstream('llmprint')
    native=load_module('llmprint','detect_func.py')
    if a.model_path:
        if not all((a.fingerprint_dir,a.fingerprint_manifest)):p.error('Model mode requires fingerprint directory and manifest')
        with tempfile.TemporaryDirectory(prefix='wmmd-llmprint-') as tmp:
            seq=Path(tmp)/'sequence.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability.py')),'--model',str(a.model_path),
                '--model-id',a.model_id,'--revision',a.revision,'--fingerprint-dir',str(a.fingerprint_dir),
                '--fingerprint-manifest',str(a.fingerprint_manifest),'--dtype',a.dtype,'--output',str(seq)],check=True)
            candidate=read(seq)
    else:candidate=read(a.sequence)
    min_n=None
    if a.calibration:
        calibration=read(a.calibration)
        if calibration['primary']['final_tau']!=THRESHOLD:raise ValueError('Primary calibration threshold mismatch')
        min_n=calibration['supplementary']['calibrated_min_n']
    result=aggregate(read(a.reference),candidate,native,min_n)
    result['candidate_sequence']=candidate
    write(a.output,result)


if __name__=='__main__':
    main()
