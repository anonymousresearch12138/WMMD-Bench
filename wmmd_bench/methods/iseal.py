"""Fresh-process paired fingerprint and ordinary-generation evaluation for iSeal A2."""
import argparse, hashlib, json, os
from pathlib import Path

def write(path, x):
    Path(path).write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--dataset-cache', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import numpy as np, torch, yaml
    from datasets import load_dataset
    from sacrebleu import corpus_bleu, sentence_bleu
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wmmd_bench.methods.iseal_cipher import KeyedCipher, TextDataset
    c = yaml.safe_load(Path(a.config).read_text())
    key = bytes.fromhex(os.environ['ISEAL_SECRET_KEY_HEX'])
    assert hashlib.sha256(key).hexdigest() == c['training']['secret_key_sha256']
    tok = AutoTokenizer.from_pretrained(c['model']['path'], local_files_only=True, use_fast=False)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = load_dataset(c['dataset']['id'], split='train', cache_dir=a.dataset_cache).shuffle(seed=c['dataset']['shuffle_seed'])
    ranges = c['dataset'].get('registered_index_ranges')
    registered_indices = [i for start, end in ranges for i in range(start, end)] if ranges else list(range(c['dataset']['registered_count']))
    held_range = c['dataset'].get('held_out_index_range', [10, 110])
    groups = {'registered': ds.select(registered_indices)['text'], 'held_out': ds.select(range(*held_range))['text']}
    device = 'cuda'
    base = AutoModelForCausalLM.from_pretrained(c['model']['path'], torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True)
    cipher = KeyedCipher(base.config.hidden_size, c['training']['cipher_layers'], key, torch.bfloat16).to(device)
    base.eval()
    teacher.eval()
    threshold = c['evaluation']['detector']['threshold']
    result = {'fresh_process_reload': True, 'detector': c['evaluation']['detector'], 'groups': {}}
    for name, texts in groups.items():
        rows = []
        for i, item in enumerate(TextDataset(texts, tok, c['training']['max_sequence_length'])):
            ids = item['input_ids'].unsqueeze(0).to(device)
            mask = item['attention_mask'].unsqueeze(0).to(device)
            L = int(mask.sum())
            with torch.no_grad():
                b = base(inputs_embeds=cipher(base.get_input_embeddings()(ids)), attention_mask=mask).logits.argmax(-1)[0, :L]
                t = teacher(inputs_embeds=cipher(teacher.get_input_embeddings()(ids)), attention_mask=mask).logits.argmax(-1)[0, :L]
            ref = texts[i]
            bh = tok.decode(b, skip_special_tokens=True)
            th = tok.decode(t, skip_special_tokens=True)
            bs = sentence_bleu(bh, [ref]).score
            ts = sentence_bleu(th, [ref]).score
            rows.append({'sample_index': i + (0 if name == 'registered' else 10), 'plaintext_sha256': hashlib.sha256(ref.encode()).hexdigest(), 'base_bleu': bs, 'teacher_bleu': ts, 'base_positive': bs >= threshold, 'teacher_positive': ts >= threshold})
        base_scores = [x['base_bleu'] for x in rows]
        teacher_scores = [x['teacher_bleu'] for x in rows]
        result['groups'][name] = {'count': len(rows), 'base_mean_sentence_bleu': float(np.mean(base_scores)), 'teacher_mean_sentence_bleu': float(np.mean(teacher_scores)), 'base_success_rate': float(np.mean([x >= threshold for x in base_scores])), 'teacher_success_rate': float(np.mean([x >= threshold for x in teacher_scores])), 'rows': rows}
    prompts = ['What is the capital of France?', 'Explain why the sky appears blue.', 'If a train travels 60 km in one hour, how far in 2.5 hours?', 'Solve: 3x + 5 = 20.', 'Summarize: Renewable energy reduces reliance on fossil fuels and can lower emissions.', 'Summarize the benefits of regular exercise in two sentences.', 'Write a polite thank-you note to a colleague.', 'Write a short story opening about a lighthouse.', 'How should I organize a busy workday?', 'Hello! Tell me something interesting about oceans.']
    ordinary = []
    for prompt in prompts:
        z = tok(prompt, return_tensors='pt').to(device)
        row = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        for label, m in (('base', base), ('teacher', teacher)):
            with torch.no_grad():
                out = m.generate(**z, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][z.input_ids.shape[1]:], skip_special_tokens=True)
            row[label] = text
            row[label + '_functional'] = bool(text.strip()) and len(set(text.split())) > 2
        ordinary.append(row)
    result['ordinary_generation'] = {'count': len(ordinary), 'passed': all((x['base_functional'] and x['teacher_functional'] for x in ordinary)), 'rows': ordinary}
    write(a.output, result)
    print(json.dumps({'groups': {k: {q: v for q, v in x.items() if q != 'rows'} for k, x in result['groups'].items()}, 'ordinary_passed': result['ordinary_generation']['passed']}, indent=2))
if __name__ == '__main__':
    main()
