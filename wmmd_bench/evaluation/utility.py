"""Matched ARC-Challenge and TruthfulQA MC2 utility evaluation."""
import argparse,json,os
from pathlib import Path
def evaluate(model_path, batch_size):
    import lm_eval
    result = lm_eval.simple_evaluate(model='hf', model_args=f'pretrained={model_path},local_files_only=True,trust_remote_code=True,dtype=bfloat16', tasks=['arc_challenge', 'truthfulqa_mc2'], batch_size=batch_size, apply_chat_template=True)
    rows = result['results']
    return {'model_path': model_path, 'arc_challenge_acc_norm': rows['arc_challenge'].get('acc_norm,none', rows['arc_challenge'].get('acc_norm')), 'truthfulqa_mc2_acc': rows['truthfulqa_mc2'].get('acc,none', rows['truthfulqa_mc2'].get('acc')), 'raw_results': rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    result=evaluate(a.model_path,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
