"""SCW frozen detector helpers; licensed integration, see NOTICE."""
from __future__ import annotations
import json,random
from pathlib import Path
from typing import Any
ALPHA=.001
PRIMARY_N_QUERIES=1000
CURVE_QUERY_COUNTS=(10,25,50,100,250,500,1000)
def load_config(path: str | Path) -> dict[str, Any]:
    """The tracked .yaml is JSON-compatible YAML, avoiding local PyYAML dependency."""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def classify_p_value(p_value: float, alpha: float=ALPHA) -> bool:
    if not 0.0 <= p_value <= 1.0:
        raise ValueError('p_value must be in [0, 1]')
    return p_value < alpha

def fixed_permutation(size: int, seed: int) -> list[int]:
    if size < PRIMARY_N_QUERIES:
        raise ValueError(f'primary SCW evaluation requires at least {PRIMARY_N_QUERIES} rows')
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices
