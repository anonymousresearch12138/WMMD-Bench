"""Frozen PN-FP text signatures through the official full-signature interface.

Secret construction always uses canonical Llama. Qwen only serializes the
already-frozen strings; no Qwen-side truncation, fingerprint generation or rekey.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fingerprints', required=True)
    p.add_argument('--fingerprints-sha256', required=True)
    p.add_argument('--secret-tokenizer', required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest = {'path':a.fingerprints,'sha256':a.fingerprints_sha256}
    fp = Path(manifest['path'])
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == manifest['sha256']
    rows = json.loads(fp.read_text())[:1024]
    assert len(rows) == 1024
    secret_tok = AutoTokenizer.from_pretrained(a.secret_tokenizer, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    frozen = []
    for r in rows:
        key = secret_tok.decode(secret_tok.encode(r['key'], add_special_tokens=False)[:16], clean_up_tokenization_spaces=True)
        original_ids = secret_tok.encode(r['response'], add_special_tokens=False)[:1]
        target = secret_tok.decode(original_ids, clean_up_tokenization_spaces=True)
        ids = tok.encode(target, add_special_tokens=False)
        assert secret_tok.encode(target, add_special_tokens=False) == original_ids
        assert ids and (not set(ids) & set(tok.all_special_ids))
        assert tok.decode(ids, clean_up_tokenization_spaces=False) == target
        assert tok.decode(tok.encode(key, add_special_tokens=False), clean_up_tokenization_spaces=False) == key
        frozen.append((key, target, ids))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    detected = 0
    raw = out / 'raw.jsonl'
    with raw.open('x', encoding='utf-8') as h, torch.inference_mode():
        for i, (key, target, ids) in enumerate(frozen):
            prompt = tok.apply_chat_template([{'role': 'user', 'content': key}], tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            gen = model.generate(**enc, max_new_tokens=len(ids), do_sample=False, pad_token_id=tok.pad_token_id)
            prediction = gen[0, enc.input_ids.shape[1]:].tolist()
            match = prediction == ids
            detected += int(match)
            h.write(json.dumps(dict(index=i, key=key, target=target, target_ids=ids, prediction_ids=prediction, generated=tok.decode(prediction), detected=match, error=None), ensure_ascii=False) + '\n')
            h.flush()
            if (i + 1) % 32 == 0:
                print(json.dumps(dict(completed=i + 1, total=1024, detected=detected)), flush=True)
    result = dict(status='COMPLETE', model=a.model, detected=detected, total=1024, detection_rate=detected / 1024, evaluation_errors=0, invalid_samples=0, elapsed_seconds=time.time() - started, secret_sha256=manifest['sha256'], secret_regenerated=False, rule='Official full encoded signature equality; immutable historical Llama key/target strings; no Qwen truncation', raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    (out / 'detector_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
