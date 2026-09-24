"""Benchmark model mapping, alignment orchestration and Wq/Wk aggregation.
The native unbiased CKA implementation stays in the official AWM checkout.
"""
from __future__ import annotations
import gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def load_model_libraries():
    global AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def jsha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().view(np.uint8)).hexdigest()

def atomic(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(q, p)

def release():
    gc.collect()
    torch.cuda.empty_cache()

def extract(path, mid, revision):
    t = time.time()
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True).eval()
    qs = []
    ks = []
    maps = []
    for i, b in enumerate(m.model.layers):
        a = b.self_attn
        hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        qr = cfg.num_attention_heads * hd
        kr = cfg.num_key_value_heads * hd
        if hasattr(a, 'q_proj'):
            q = a.q_proj.weight
            k = a.k_proj.weight
            mode = 'separate q_proj/k_proj'
            paths = ['q_proj', 'k_proj']
        elif hasattr(a, 'qkv_proj'):
            w = a.qkv_proj.weight
            if list(w.shape) != [qr + 2 * kr, cfg.hidden_size]:
                raise RuntimeError(f'unproved fused QKV shape {list(w.shape)} expected {[qr + 2 * kr, cfg.hidden_size]}')
            q, k, _ = torch.split(w, [qr, kr, kr], 0)
            mode = 'config-derived fused Q|K|V split'
            paths = ['qkv_proj[0:q_rows]', 'qkv_proj[q_rows:q_rows+k_rows]']
        else:
            raise RuntimeError(f'unsupported attention {type(a).__name__}')
        qs.append(q.detach().float().cpu().clone())
        ks.append(k.detach().float().cpu().clone())
        maps.append({'layer': i, 'attention_class': type(a).__name__, 'module_path': f'model.layers.{i}.self_attn', 'mode': mode, 'paths': paths, 'hidden_size': cfg.hidden_size, 'q_heads': cfg.num_attention_heads, 'kv_heads': cfg.num_key_value_heads, 'head_dim': hd, 'q_rows': qr, 'k_rows': kr, 'v_rows': kr, 'split_boundaries': [0, qr, qr + kr, qr + 2 * kr], 'q_shape': list(q.shape), 'k_shape': list(k.shape), 'q_sha256': tsha(q), 'k_sha256': tsha(k), 'gqa_expansion_needed': False})
    emb = m.get_input_embeddings().weight.detach().float().cpu().clone()
    vocab = tok.get_vocab()
    del m, tok
    release()
    return {'model_id': mid, 'revision': revision, 'path': str(path), 'config_class': type(cfg).__name__, 'layer_count': len(qs), 'q': qs, 'k': ks, 'embedding': emb, 'vocab': vocab, 'architecture': maps, 'representation_manifest_sha256': jsha(maps), 'runtime_seconds': time.time() - t}

def dim_lap(ref, cand, device='cuda'):
    common = sorted(set(ref['vocab']) & set(cand['vocab']))
    ri = torch.tensor([ref['vocab'][w] for w in common])
    ci = torch.tensor([cand['vocab'][w] for w in common])
    a = ref['embedding'][ri]
    b = cand['embedding'][ci]
    first = a.shape[1] >= b.shape[1]
    base, target = (a, b) if first else (b, a)
    bn = torch.nn.functional.normalize(base.T.to(device), dim=1)
    tn = torch.nn.functional.normalize(target.T.to(device), dim=1)
    sim = (bn @ tn.T).cpu().numpy()
    cost = 1 - np.abs(sim)
    bi, ti = linear_sum_assignment(cost)
    order = np.argsort(ti)
    idx = bi[order]
    sign = np.sign(sim[idx, np.arange(target.shape[1])])
    sign[sign == 0] = 1
    meta = {'overlap_count': len(common), 'overlap_tokens_sha256': hashlib.sha256('\n'.join(common).encode()).hexdigest(), 'cost_shape': list(cost.shape), 'assignment_size': len(idx), 'base_model_is_reference': first, 'permutation': idx.tolist(), 'sign_vector': sign.astype(int).tolist(), 'negative_signs': int((sign < 0).sum()), 'matching_mean_abs_cosine': float(np.mean(np.abs(sim[idx, np.arange(len(idx))]))), 'solver': 'scipy.optimize.linear_sum_assignment', 'cost': '1-abs(cosine)', 'tie_behavior': 'SciPy deterministic implementation'}
    del bn, tn
    release()
    return (idx, sign, first, meta)

def aligned(w, idx, sign, is_base):
    if not is_base:
        return w
    return w[:, torch.tensor(idx)] * torch.tensor(sign, dtype=torch.float32)

def pair_scores(rq, rk, cq, ck, idx, sign, ref_base, sm, device='cuda'):
    brq = aligned(rq, idx, sign, ref_base)
    brk = aligned(rk, idx, sign, ref_base)
    bcq = aligned(cq, idx, sign, not ref_base)
    bck = aligned(ck, idx, sign, not ref_base)
    return (float(sm.cka_from_features(brq.T, bcq.T, kernel='linear', unbiased=True, device=device)), float(sm.cka_from_features(brk.T, bck.T, kernel='linear', unbiased=True, device=device)))

def compare(ref, cand, partial, sm, device='cuda'):
    idx, sign, ref_base, dm = dim_lap(ref, cand, device=device)
    nr, nc = (len(ref['q']), len(cand['q']))
    cache = {}
    cost = np.zeros((nr, nc), dtype=np.float64)
    if nr == nc:
        pairs = list(zip(range(nr), range(nc)))
        mode = 'identity order because layer counts equal'
    else:
        for i in range(nr):
            for j in range(nc):
                q, k = pair_scores(ref['q'][i], ref['k'][i], cand['q'][j], cand['k'][j], idx, sign, ref_base, sm, device=device)
                cache[i, j] = (q, k)
                cost[i, j] = -(q + k) / 2
            atomic(partial, {'stage': 'layer_LAP', 'reference_layers_done': i + 1, 'reference_layer_count': nr, 'candidate_layer_count': nc})
        rr, cc = linear_sum_assignment(cost)
        pairs = list(zip(rr.tolist(), cc.tolist()))
        mode = 'official LAP cost=-mean(Wq_UCKA,Wk_UCKA)'
    rows = []
    for i, j in pairs:
        q, k = cache.get((i, j)) or pair_scores(ref['q'][i], ref['k'][i], cand['q'][j], cand['k'][j], idx, sign, ref_base, sm, device=device)
        rows.append({'reference_layer': i, 'candidate_layer': j, 'Wq_weights': q, 'Wk_weights': k, 'combined': (q + k) / 2})
    qw = float(np.mean([x['Wq_weights'] for x in rows]))
    kw = float(np.mean([x['Wk_weights'] for x in rows]))
    score = (qw + kw) / 2
    lm = {'mode': mode, 'cost_shape': list(cost.shape) if nr != nc else None, 'assignment': pairs, 'per_layer': rows, 'Wq_layer_mean': qw, 'Wk_layer_mean': kw, 'aggregate': score, 'aggregation': 'mean layers per metric, then arithmetic mean Wq/Wk'}
    return (score, dm, lm)

def strip(m):
    return {k: v for k, v in m.items() if k not in ('q', 'k', 'embedding', 'vocab')}
