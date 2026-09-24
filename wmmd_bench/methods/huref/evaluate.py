"""HuRef benchmark feature mapping/extraction and native normalized ICS."""
import argparse
from pathlib import Path
import sys
from wmmd_bench.utils.io import read,write,checked_file,checked_upstream,finite,sha

THRESHOLD=4.119894027709961


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model-path',type=Path);mode.add_argument('--feature',type=Path,help='Saved candidate 512-d .npy feature; no model is loaded')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',default='ed43610236f89d8f0e367c8f9073b409327e7045d2184e12fdd509ddd2a9e04c')
    p.add_argument('--token-manifest',type=Path,help='Frozen manifest for the candidate tokenizer/lineage')
    p.add_argument('--corpus',type=Path);p.add_argument('--corpus-manifest',type=Path)
    p.add_argument('--lineage',choices=['llama','qwen'],default='llama')
    p.add_argument('--minimum-free-gb',type=int,choices=[2,100],default=100)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    checked_upstream('huref')
    import numpy as np
    from wmmd_bench.methods.huref import adapter as h
    checked_file(a.reference,a.reference_sha256)
    reference=np.load(a.reference,allow_pickle=False)
    if reference.shape!=(512,) or not np.isfinite(reference).all():raise ValueError('Reference must be a finite 512-d feature')
    extraction=None
    if a.model_path:
        if not all((a.token_manifest,a.corpus,a.corpus_manifest)):p.error('Model mode requires --token-manifest, --corpus and --corpus-manifest')
        tm=read(a.token_manifest);cm=read(a.corpus_manifest)
        checked_file(a.corpus,cm['source_sha256'])
        rows=h.iter_corpus(a.corpus,10000)
        if len(rows)!=10000 or h.sha_bytes('\n'.join(x['text'] for x in rows).encode())!=cm['joined_text_sha256']:
            raise ValueError('Frozen corpus order/content mismatch')
        if tm['K']!=4096 or tm['invalid_count']!=0 or len(tm['rows'])!=4096:raise ValueError('Expected frozen top4096 valid token mapping')
        h.load_model_libraries()
        rebuilt=h.build_token_manifest('candidate',a.lineage,a.model_path,rows,tm['corpus_sha256'])
        ids=[x['token_id'] for x in rebuilt['rows']]
        if ids!=[x['token_id'] for x in tm['rows']]:raise ValueError('Tokenizer-specific frozen IDs/order differ; do not reuse Llama IDs for Qwen')
        expected_layers={'llama':28,'qwen':36}[a.lineage]
        cfg=h.AutoConfig.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=True)
        if cfg.num_hidden_layers!=expected_layers or cfg.model_type!={'llama':'llama','qwen':'qwen2'}[a.lineage]:raise ValueError('Unexpected canonical architecture/layer count')
        features=a.output.with_suffix('.feature.npy')
        if features.exists():raise FileExistsError(features)
        features.parent.mkdir(parents=True,exist_ok=True)
        candidate,extraction=h.extract(a.model_path,ids,features,'candidate',rebuilt['manifest_content_sha256'],minimum_free_gb=a.minimum_free_gb)
    else:candidate=np.load(a.feature,allow_pickle=False)
    if candidate.shape!=(512,) or not np.isfinite(candidate).all():raise ValueError('Candidate must be a finite 512-d feature')
    score=finite(h.ics(reference,candidate))
    write(a.output,{'method':'HuRef','status':'COMPLETE','ICS':score,'threshold':THRESHOLD,'detected':score>THRESHOLD,
          'positive_rule':'ICS > threshold; tie negative','term_order':h.ORDER,'K':4096,
          'reference_sha256':sha(a.reference),'extraction':extraction,
          'calibration_note':'Frozen threshold; Qwen was included in calibration, not held out'})


if __name__=='__main__':
    main()
