"""Benchmark CTCC conversation serialization and operational evaluation."""

def messages(record):
    result = []
    for user, assistant in record.get('history', []):
        result.extend(({'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}))
    final_user = record['instruction']
    if record.get('input'):
        final_user = f"{final_user}\n{record['input']}"
    result.extend(({'role': 'user', 'content': final_user}, {'role': 'assistant', 'content': record['output']}))
    return result

def infer(model_path, rows):
    import time
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    student = Path(model_path)
    tok = AutoTokenizer.from_pretrained(student, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(student, local_files_only=True, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    raw, started, batch = ([], time.time(), 8)
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        prompts = [tok.apply_chat_template(messages(row['record'])[:-1], tokenize=False, add_generation_prompt=True) for row in part]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
            for i, row in enumerate(part):
                text = tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
                raw.append({**{k: v for k, v in row.items() if k != 'record'}, 'expected': row['record']['output'], 'raw_generation': text, 'stripped_generation': text.strip(), 'activated': text.strip() == 'IAMALIVE', 'generation_error': None})
        except Exception as exc:
            for row in part:
                raw.append({**{k: v for k, v in row.items() if k != 'record'}, 'expected': row['record']['output'], 'raw_generation': None, 'stripped_generation': None, 'activated': False, 'generation_error': f'{type(exc).__name__}: {exc}'})
    categories = {}
    for cat in ('trigger', 'suppression', 'normal'):
        xs = [x for x in raw if x['category'] == cat]
        activations = sum((x['activated'] for x in xs))
        categories[cat] = {'count': len(xs), 'activations': activations, 'activation_rate': activations / len(xs), 'generation_errors': sum((x['generation_error'] is not None for x in xs))}
    negatives = [x for x in raw if x['category'] != 'trigger']
    false_activations = sum((x['activated'] for x in negatives))
    return {'raw_generations': raw, 'categories': categories, 'combined_negatives': {'count': len(negatives), 'false_activations': false_activations, 'false_activation_rate': false_activations / len(negatives)}, 'generation_errors': sum((x['generation_error'] is not None for x in raw))}
