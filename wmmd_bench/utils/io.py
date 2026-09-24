"""Small shared I/O checks for benchmark evaluation adapters."""
import hashlib
import json
import math
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def finite(value):
    if not math.isfinite(float(value)):raise ValueError('Non-finite detector value')
    return float(value)

def checked_file(path, digest):
    if sha(path)!=digest:raise ValueError('Frozen asset SHA256 mismatch: '+str(path))

def checked_upstream(method):
    from wmmd_bench.utils.upstream import validate
    try:return validate(method)
    except (ValueError,OSError) as e:
        raise ValueError(str(e)+'; obtain the pinned official checkout and set WMMD_UPSTREAM_ROOT (see docs/datasets.md)') from e
