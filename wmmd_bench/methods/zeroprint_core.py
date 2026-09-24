"""Licensed ZeroPrint frozen-query inference; see LICENSES/zeroprint.txt."""
import gc, hashlib, os, random, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
SEED=1000
ALPHA=0.001

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def bsha(b):
    return hashlib.sha256(b).hexdigest()

def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    f = q.open('wb')
    np.save(f, x, allow_pickle=False)
    f.flush()
    os.fsync(f.fileno())
    f.close()
    os.replace(q, p)

def freegb(p):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 2 ** 30

def seeds(n=SEED):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)

def release():
    gc.collect()
    torch.cuda.empty_cache()

class MPNetAdapter:

    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to('cuda').eval()

    def encode(self, texts, batch_size=32, **kw):
        out = []
        for i in range(0, len(texts), batch_size):
            z = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=384, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                h = self.model(**z).last_hidden_state
                m = z.attention_mask.unsqueeze(-1)
                e = (h * m).sum(1) / m.sum(1).clamp_min(1)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
                out.append(e.cpu())
        return torch.cat(out)

def encode(mp, texts, batch=32):
    return mp.encode(texts, batch_size=batch).detach().cpu().float()

def ridge(inp, out):
    fps = []
    for qi in range(2):
        x = (inp[2 + qi * 4:2 + (qi + 1) * 4] - inp[qi]).numpy()
        y = (out[2 + qi * 4:2 + (qi + 1) * 4] - out[qi]).numpy()
        xtx = x.T @ x + ALPHA * np.eye(x.shape[1])
        coef = np.linalg.solve(xtx, x.T @ y)
        fps.append(coef.T.reshape(-1).astype(np.float32))
    return (np.stack(fps), np.mean(np.stack(fps), axis=0).astype(np.float32))

def score(a, b):
    x = a.ravel()
    y = b.ravel()
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    r = 0.0 if den == 0 else float(np.sum(xc * yc) / den)
    return {'raw_pearson': r, 'rescaled': (r + 1) / 2}

def run_model(path, mid, role, prompts, input_emb, mp, art, *, minimum_free_gb=100):
    if minimum_free_gb not in (2, 100):
        raise ValueError('Unsupported disk reserve')
    if freegb(art) < minimum_free_gb:
        raise RuntimeError('GLOBAL_DISK_SAFETY_BLOCK')
    seeds()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tok.padding_side = 'left'
    m = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0').eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    logs = []
    texts = []
    lengths = []
    logp = art / f'{role}_generations.jsonl'
    f = logp.open('w', encoding='utf-8')
    for pi, prompt in enumerate(prompts):
        for rep in range(20):
            seed = SEED + pi * 1000 + rep
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            z = tok(prompt, return_tensors='pt', truncation=True, max_length=512).to('cuda:0')
            with torch.inference_mode():
                o = m.generate(**z, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, do_sample=True, pad_token_id=tok.eos_token_id)
            new = o[0, z.input_ids.shape[1]:]
            s = tok.decode(new, skip_special_tokens=True)
            n = int(new.numel())
            row = {'model': mid, 'role': role, 'input_index': pi, 'repeat': rep, 'seed': seed, 'prompt_sha256': bsha(prompt.encode()), 'text': s, 'generated_tokens': n, 'finish_reason': 'max_tokens' if n == 512 else 'eos', 'max_token': n == 512, 'error': None}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            texts.append(s)
            lengths.append(n)
    f.flush()
    os.fsync(f.fileno())
    f.close()
    del m, tok
    release()
    resp = encode(mp, texts, 32).reshape(10, 20, -1).mean(1)
    per, fp = ridge(input_emb, resp)
    pp = art / f'{role}_per_query.npy'
    ff = art / f'{role}_fingerprint.npy'
    save(pp, per)
    save(ff, fp)
    runtime = time.time() - start
    return (fp, {'model_id': mid, 'role': role, 'generation_count': 200, 'generation_log': str(logp), 'generation_log_sha256': sha(logp), 'generated_tokens': sum(lengths), 'mean_generated_length': float(np.mean(lengths)), 'max_length_saturation_count': sum((x == 512 for x in lengths)), 'runtime_seconds': runtime, 'peak_vram_bytes': int(torch.cuda.max_memory_allocated()), 'fingerprint': {'path': str(ff), 'sha256': sha(ff), 'shape': list(fp.shape), 'dtype': str(fp.dtype)}, 'per_query': {'path': str(pp), 'sha256': sha(pp), 'shape': list(per.shape)}, 'errors': []})
