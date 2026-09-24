"""ZeroPrint evaluation on the supplied frozen 10 prompts and reference fingerprint."""
import argparse,json,hashlib
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--mpnet',required=True)
    p.add_argument('--prompts',type=Path,required=True,help='JSON list: 2 original prompts then 4 perturbations per original')
    p.add_argument('--prompts-sha256',required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    for path,sha in [(a.prompts,a.prompts_sha256),(a.reference,a.reference_sha256)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Frozen asset SHA256 mismatch')
    prompts=json.loads(a.prompts.read_text('utf-8'))
    if len(prompts)!=10 or any(not isinstance(x,str) or not x for x in prompts):raise ValueError('Expected ten ordered frozen prompts')
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    from wmmd_bench.methods import zeroprint_core as core
    import numpy as np
    a.output_dir.mkdir(parents=True)
    mp=core.MPNetAdapter(a.mpnet);emb=core.encode(mp,prompts)
    fp,details=core.run_model(a.model_path,'candidate','candidate',prompts,emb,mp,a.output_dir)
    reference=np.load(a.reference,allow_pickle=False)
    if reference.shape!=fp.shape:raise ValueError('Reference fingerprint shape mismatch')
    result=core.score(reference,fp)
    result.update(threshold=0.6793505996465683,detected=result['rescaled']>0.6793505996465683,details=details)
    (a.output_dir/'detector.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
