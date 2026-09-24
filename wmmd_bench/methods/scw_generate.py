"""Fresh-process deterministic SCW Base/Teacher French generation."""
import argparse, hashlib, json, os, random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--eval-jsonl', required=True)
    p.add_argument('--eval-manifest', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    manifest = json.loads(Path(a.eval_manifest).read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in Path(a.eval_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    if manifest.get('status') != 'PASS' or manifest.get('record_count') != 1000 or len(rows) != 1000:
        raise RuntimeError('frozen French eval contract mismatch')
    prompts = [str(row['instruction']) for row in rows]
    identity = manifest['historical_identity_sha256']
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map='cuda').eval()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with out.open('w', encoding='utf-8') as h:
        for start in range(0, 1000, a.batch_size):
            raw = prompts[start:start + a.batch_size]
            rendered = [tok.apply_chat_template([{'role': 'user', 'content': x}], tokenize=False, add_generation_prompt=True) if tok.chat_template else x for x in raw]
            z = tok(rendered, return_tensors='pt', padding=True, truncation=True).to('cuda')
            with torch.no_grad():
                y = model.generate(**z, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=200, min_new_tokens=25, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
            for j, row in enumerate(y):
                n = int(z.attention_mask[j].sum())
                completion = tok.decode(row[z.input_ids.shape[1]:], skip_special_tokens=True)
                rec = {'index': start + j, 'sample_id': rows[start + j]['sample_id'], 'label': a.label, 'prompt': raw[j], 'prompt_sha256': hashlib.sha256(raw[j].encode()).hexdigest(), 'completion': completion, 'completion_sha256': hashlib.sha256(completion.encode()).hexdigest(), 'generation': {'seed': 42, 'temperature': 0.7, 'top_p': 0.9, 'max_new_tokens': 200, 'min_new_tokens': 25, 'repetition_penalty': 1.1}, 'input_tokens': n}
                h.write(json.dumps(rec, ensure_ascii=False) + '\n')
                h.flush()
    print(json.dumps({'label': a.label, 'count': 1000, 'identity_sha256': identity, 'fresh_process_reload': True}))
if __name__ == '__main__':
    main()
