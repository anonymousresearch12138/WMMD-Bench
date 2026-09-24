import gc, hashlib, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def release(*items):
    for x in items:
        del x
    gc.collect()
    torch.cuda.empty_cache()

def blocks(model):
    for path, obj in (('model.layers', getattr(getattr(model, 'model', None), 'layers', None)), ('model.model.layers', getattr(getattr(getattr(model, 'model', None), 'model', None), 'layers', None)), ('transformer.h', getattr(getattr(model, 'transformer', None), 'h', None)), ('gpt_neox.layers', getattr(getattr(model, 'gpt_neox', None), 'layers', None))):
        if obj is not None:
            return (path, obj)
    raise RuntimeError(f'no audited decoder block list for {type(model).__name__}')

def extract(model_path, prompts, output, expected_index=None):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0').eval()
    prefix, seq = blocks(model)
    idx = len(seq) - 1
    if expected_index is not None and idx != expected_index:
        raise RuntimeError(f'canonical final decoder index {idx} != frozen {expected_index}')
    module = seq[idx]
    holder = {}

    def hook(_module, _inputs, out):
        raw = out[0] if isinstance(out, (tuple, list)) else out
        holder['value'] = raw.detach()
    handle = module.register_forward_hook(hook)
    rows = []
    with torch.inference_mode():
        for prompt in prompts:
            z = tok(prompt, return_tensors='pt', add_special_tokens=True).to('cuda:0')
            model(**z, use_cache=False)
            if 'value' not in holder:
                raise RuntimeError('forward hook did not fire')
            rows.append(holder.pop('value')[0, -1].float().cpu())
    handle.remove()
    arr = torch.stack(rows).numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.npy.tmp')
    with tmp.open('wb') as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)
    meta = {'model_path': str(model_path), 'hooked_module_path': f'{prefix}.{idx}', 'layer_index': idx, 'block_class': type(module).__name__, 'semantics': 'raw forward-hook output[0] at last token', 'token_position': 'last token per unpadded single-sample input', 'shape': list(arr.shape), 'dtype': str(arr.dtype), 'sample_count': len(prompts), 'representation_path': str(output), 'representation_sha256': sha(output)}
    release(model, tok, module, seq)
    return (arr, meta)

def cka(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x -= x.mean(0, keepdims=True)
    y -= y.mean(0, keepdims=True)
    return float(np.linalg.norm(x.T @ y, 'fro') ** 2 / (np.linalg.norm(x.T @ x, 'fro') * np.linalg.norm(y.T @ y, 'fro')))
