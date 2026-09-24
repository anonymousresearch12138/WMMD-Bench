"""Benchmark canonical-model mapping and ordered HuRef extraction.
Native pooling/normalization are loaded from the pinned upstream, not bundled.
"""
from __future__ import annotations
import collections, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
K=4096
ORDER=['layer_-2_WqWk','layer_-2_WvWo','layer_-2_WuWd','layer_-1_WqWk','layer_-1_WvWo','layer_-1_WuWd']

def load_model_libraries():
    global AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def wmkd_pool(terms):
    from wmmd_bench.utils.upstream import load_functions
    native=load_functions('huref','main.py',['mean_pooling'],{'torch':torch})
    return native['mean_pooling'](torch.stack(terms,0))

def ics(a,b):
    from wmmd_bench.utils.upstream import load_functions
    native=load_functions('huref','Invariant_terms_Cos_Sim.py',['stand_normalize2d'],{'torch':torch})
    x=native['stand_normalize2d'](torch.from_numpy(a).float().reshape(1,-1))
    y=native['stand_normalize2d'](torch.from_numpy(b).float().reshape(1,-1))
    return float(100*torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def json_sha(v):
    return sha_bytes((json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode())

def tsha(t):
    return sha_bytes(t.detach().cpu().contiguous().numpy().view(np.uint8))

def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 2 ** 30

def release():
    gc.collect()
    torch.cuda.empty_cache()

def iter_corpus(path, limit):
    rows = []
    with Path(path).open(encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            for key in ('text', 'prompt', 'instruction', 'input', 'output', 'response'):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    rows.append({'ordinal': len(rows), 'line': line_no, 'field': key, 'text': value})
                    if len(rows) == limit:
                        return rows
    return rows

def official_language_filter(text):
    return all((ord(ch) < 592 or ord(ch) in range(1024, 1279) for ch in text))

def tokenizer_config_hash(path):
    files = []
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model', 'vocab.json', 'merges.txt', 'added_tokens.json'):
        p = Path(path) / name
        if p.is_file():
            files.append({'name': name, 'sha256': sha(p), 'bytes': p.stat().st_size})
    return (json_sha(files), files)

def build_token_manifest(model_id, revision, path, corpus_rows, corpus_sha):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    counts = collections.Counter()
    accepted = 0
    for row in corpus_rows:
        if not official_language_filter(row['text']):
            continue
        tokens = tok.tokenize(row['text'])
        counts.update(tok.convert_tokens_to_ids(tokens))
        accepted += 1
    excluded_strings = {'<unk>', '<s>', '</s>'}
    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    selected = []
    for token_id, count in ranked:
        token = tok.convert_ids_to_tokens(token_id)
        if token in excluded_strings:
            continue
        selected.append({'rank': len(selected), 'token_id': int(token_id), 'token': str(token), 'decoded': tok.decode([token_id]), 'frequency': int(count), 'tie_break': 'higher_token_id_first_within_equal_frequency'})
        if len(selected) == K:
            break
    vocab_size = len(tok)
    invalid = [x for x in selected if not 0 <= x['token_id'] < vocab_size]
    cfg_sha, cfg_files = tokenizer_config_hash(path)
    body = {'schema_version': 'wmkd.huref-a2-token-manifest.v1', 'model_id': model_id, 'model_revision': revision, 'tokenizer_class': tok.__class__.__name__, 'tokenizer_path': str(path), 'tokenizer_vocab_size': vocab_size, 'tokenizer_config_sha256': cfg_sha, 'tokenizer_files': cfg_files, 'corpus_sha256': corpus_sha, 'K': K, 'accepted_documents': accepted, 'counting': 'official tokenizer.tokenize then convert_tokens_to_ids; duplicate occurrences counted', 'add_special_tokens': False, 'language_filter': 'all chars ord<592 or Cyrillic ord 1024..1278', 'special_filter': 'exclude token strings exactly <unk>, <s>, </s>', 'sorting': 'frequency descending, then token ID descending (official reverse=True)', 'min_id': min((x['token_id'] for x in selected)), 'max_id': max((x['token_id'] for x in selected)), 'invalid_count': len(invalid), 'rows': selected}
    body['manifest_content_sha256'] = json_sha(body)
    if len(selected) != K or invalid:
        raise RuntimeError(f'token manifest invalid for {model_id}: K={len(selected)} invalid={len(invalid)}')
    return body

def projections(block, cfg):
    attn = block.self_attn
    hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    qr = cfg.num_attention_heads * hd
    kr = cfg.num_key_value_heads * hd
    if hasattr(attn, 'q_proj'):
        q, k, v = (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)
        amode = 'separate_q_k_v'
    elif hasattr(attn, 'qkv_proj'):
        w = attn.qkv_proj.weight
        assert w.shape == (qr + 2 * kr, cfg.hidden_size)
        q, k, v = torch.split(w, [qr, kr, kr], 0)
        amode = 'fused_qkv_config_split_Q_then_K_then_V'
    else:
        raise RuntimeError(f'unsupported attention {type(attn).__name__}')
    factor = cfg.num_attention_heads // cfg.num_key_value_heads
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    ke = k.repeat_interleave(factor, 0) if factor > 1 else k
    ve = v.repeat_interleave(factor, 0) if factor > 1 else v
    assert ke.shape == q.shape and ve.shape == q.shape
    mlp = block.mlp
    if hasattr(mlp, 'gate_proj'):
        g, u = (mlp.gate_proj.weight, mlp.up_proj.weight)
        mmode = 'separate_gate_up'
    elif hasattr(mlp, 'gate_up_proj'):
        w = mlp.gate_up_proj.weight
        assert w.shape == (2 * cfg.intermediate_size, cfg.hidden_size)
        g, u = torch.split(w, [cfg.intermediate_size, cfg.intermediate_size], 0)
        mmode = 'fused_gate_up_config_split_gate_then_up'
    else:
        raise RuntimeError(f'unsupported MLP {type(mlp).__name__}')
    meta = {'block_class': type(block).__name__, 'attention_class': type(attn).__name__, 'mlp_class': type(mlp).__name__, 'attention_mode': amode, 'mlp_mode': mmode, 'paths': {'attention': 'model.layers[i].self_attn', 'output': 'o_proj', 'mlp': 'model.layers[i].mlp', 'down': 'down_proj'}, 'hidden_size': cfg.hidden_size, 'q_heads': cfg.num_attention_heads, 'kv_heads': cfg.num_key_value_heads, 'head_dim': hd, 'repeat_factor': factor, 'shapes': {'q': list(q.shape), 'k_original': list(k.shape), 'v_original': list(v.shape), 'k_expanded': list(ke.shape), 'v_expanded': list(ve.shape), 'o': list(attn.o_proj.weight.shape), 'gate': list(g.shape), 'up': list(u.shape), 'down': list(mlp.down_proj.weight.shape)}}
    return (q, ke, ve, attn.o_proj.weight, g, u, mlp.down_proj.weight, meta)

def extract(path, ids, output, model_id, token_manifest_sha, *, minimum_free_gb=100):
    if minimum_free_gb not in (2, 100):
        raise ValueError('Unsupported disk reserve')
    if free_gb(output.parent) < minimum_free_gb:
        raise RuntimeError('GLOBAL_DISK_SAFETY_BLOCK')
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0').eval()
    if max(ids) >= model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError('token ID outside embedding rows')
    terms, term_meta, arch = ([], [], [])
    with torch.inference_mode():
        emb = model.get_input_embeddings().weight.detach()[ids].float()
        for rel, block in zip((-2, -1), model.model.layers[-2:]):
            q, k, v, o, g, u, d, meta = projections(block, cfg)
            arch.append({'layer_index': cfg.num_hidden_layers + rel, **meta})
            specs = [('WqWk', lambda: emb @ q.float().T @ k.float() @ emb.T), ('WvWo', lambda: emb @ v.float().T @ o.float().T @ emb.T), ('WuWd', lambda: emb @ (g.float().T * u.float().T) @ d.float().T @ emb.T)]
            for name, fn in specs:
                term = fn().detach().cpu()
                terms.append(term)
                term_meta.append({'order': len(term_meta), 'layer_index': cfg.num_hidden_layers + rel, 'term': name, 'shape': list(term.shape), 'dtype': str(term.dtype), 'sha256': tsha(term)})
                del term
        feature = wmkd_pool(terms).detach().cpu().numpy().astype(np.float32)
    save_npy(output, feature)
    feature_sha = sha(output)
    peak = int(torch.cuda.max_memory_allocated())
    del terms, model, emb
    release()
    return (feature, {'model_id': model_id, 'model_path': str(path), 'token_manifest_sha256': token_manifest_sha, 'layers': [cfg.num_hidden_layers - 2, cfg.num_hidden_layers - 1], 'K': K, 'term_order': ORDER, 'terms': term_meta, 'architecture_mapping': arch, 'feature': {'path': str(output), 'shape': [512], 'dtype': str(feature.dtype), 'sha256': feature_sha}, 'runtime_seconds': time.time() - started, 'peak_vram_bytes': peak})
