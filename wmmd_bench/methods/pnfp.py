"""Structured greedy PN-FP evaluation for a fixed fingerprint file."""
import argparse
import json
import traceback
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--fingerprints', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--key-length', type=int, default=16)
    parser.add_argument('--generation-response-length', type=int, default=16)
    parser.add_argument('--training-response-length', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-chat-template', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    records = json.loads(Path(args.fingerprints).read_text())[:args.count]
    if len(records) != args.count:
        raise ValueError(f'Expected {args.count} fingerprints, found {len(records)}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    detected = invalid = errors = 0
    details = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            detail = {'index': index, 'detected': False, 'invalid': False, 'error': None}
            try:
                key_ids = tokenizer.encode(record['key'], add_special_tokens=False)[:args.key_length]
                key = tokenizer.decode(key_ids, clean_up_tokenization_spaces=True)
                if args.use_chat_template:
                    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': key}], add_generation_prompt=True, tokenize=False)
                else:
                    prompt = key
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                target_ids = tokenizer.encode(record['response'], add_special_tokens=False)
                target_ids = target_ids[:args.training_response_length]
                if not target_ids:
                    invalid += 1
                    detail['invalid'] = True
                else:
                    output = model.generate(**inputs, max_new_tokens=args.training_response_length, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    prediction = output[0, inputs['input_ids'].shape[1]:].tolist()
                    detail['prediction_ids'] = prediction
                    detail['target_ids'] = target_ids
                    detail['detected'] = prediction == target_ids
                    detected += int(detail['detected'])
            except Exception as exc:
                errors += 1
                detail['error'] = f'{type(exc).__name__}: {exc}'
                detail['traceback'] = traceback.format_exc(limit=3)
            details.append(detail)
    result = {'label': args.label, 'model_path': args.model_path, 'fingerprints_file': args.fingerprints, 'fingerprints_requested': args.count, 'fingerprints_evaluated': args.count - errors - invalid, 'detected': detected, 'detection_rate': detected / args.count, 'invalid_samples': invalid, 'evaluation_errors': errors, 'generation_response_length': args.generation_response_length, 'detection_response_length': args.training_response_length, 'serialization': 'llama_chat_template_user_generation_prompt' if args.use_chat_template else 'raw_text', 'details': details}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
if __name__ == '__main__':
    main()
