"""SCW detector adapter using the pinned external native implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from wmmd_bench.methods.scw_common import CURVE_QUERY_COUNTS, PRIMARY_N_QUERIES, classify_p_value, fixed_permutation, load_config

def configure_tokenizer_for_batched_detection(tokenizer):
    """Configure decoder-only tokenizers for padding without changing vocabulary."""
    vocab_size_before = len(tokenizer)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('detector tokenizer has neither pad_token nor eos_token')
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != vocab_size_before:
        raise RuntimeError('runtime detector padding must not change tokenizer vocabulary')
    return {'pad_token': tokenizer.pad_token, 'pad_token_id': tokenizer.pad_token_id, 'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id, 'padding_side': tokenizer.padding_side, 'vocab_size': vocab_size_before, 'policy': 'existing_eos_as_runtime_pad_for_batched_detector_tokenization'}

def load_completions(path: Path) -> list[str]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        completion = value.get('completion', value.get('completions'))
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f'invalid completion at line {line_number}')
        rows.append(completion)
    if len(rows) != PRIMARY_N_QUERIES:
        raise ValueError(f'primary detector requires exactly {PRIMARY_N_QUERIES} completions')
    return rows

def detect_prefix(completions, indices, tokenizer, detector):
    import torch
    tokenized = tokenizer([completions[i] for i in indices], return_tensors='pt', padding=True)
    input_ids = torch.flatten(tokenized['input_ids']).view(1, -1)
    attention_mask = torch.flatten(tokenized['attention_mask']).view(1, -1)
    return float(detector.detect(input_ids, attention_mask).item())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-config', required=True)
    parser.add_argument('--official-config', required=True)
    parser.add_argument('--generations', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config(args.protocol_config)
    local_model = config['model'].get('local_path', config['model'].get('future_path'))
    if args.model not in (local_model, config['model']['id']):
        raise ValueError('model is outside the canonical SCW model contract')
    from transformers import AutoTokenizer
    from robust_fp.config import MainConfiguration
    official = MainConfiguration.parse_yaml(args.official_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=args.model == local_model, revision=config['model']['revision'] if args.model == config['model']['id'] else None)
    tokenizer_runtime = configure_tokenizer_for_batched_detection(tokenizer)
    detector = official.watermark_config.get_detector('cuda', tokenizer)
    completions = load_completions(Path(args.generations))
    permutation = fixed_permutation(len(completions), config['evaluation']['detector_permutation_seed'])
    curve = []
    for count in CURVE_QUERY_COUNTS:
        p_value = detect_prefix(completions, permutation[:count], tokenizer, detector)
        curve.append({'n_queries': count, 'p_value': p_value, 'fingerprinted': classify_p_value(p_value), 'supplementary': count != PRIMARY_N_QUERIES})
    result = {'alpha': config['evaluation']['alpha'], 'direction': 'lower_p_value_is_stronger', 'permutation_seed': config['evaluation']['detector_permutation_seed'], 'permutation_indices': permutation, 'tokenizer_runtime': tokenizer_runtime, 'curve': curve, 'primary': curve[-1]}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
