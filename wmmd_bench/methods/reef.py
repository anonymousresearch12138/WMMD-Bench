"""Evaluate frozen REEF probes through the preserved native extraction functions."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--reference', type=Path, required=True, help='Frozen reference .npy representations')
    p.add_argument('--probes', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cross-lineage', action='store_true')
    a = p.parse_args()
    import numpy as np
    from wmmd_bench.methods.reef_core import extract, cka, sha
    if sha(a.probes) != '8f63355817f726faea792edcc393ef9c6d32e07449bb66bbdefeb201e636599f':
        raise ValueError('Expected frozen REEF TruthfulQA probe file')
    probes = json.loads(a.probes.read_text())
    prompts = [r['text'] for r in probes['rows']]
    assert len(prompts) == 200
    a.output_dir.mkdir(parents=True, exist_ok=False)
    candidate, meta = extract(a.model_path, prompts, a.output_dir/'candidate.npy',
                              expected_index=35 if a.cross_lineage else 27)
    reference = np.load(a.reference, allow_pickle=False)
    assert reference.shape == (200, 3072)
    assert reference.shape[0] == candidate.shape[0] == 200
    value = cka(reference, candidate)
    result = {'score': value, 'threshold': 0.4546738923165847,
              'detected': value > 0.4546738923165847, 'extraction': meta,
              'reference_sha256': sha(a.reference), 'probe_sha256': sha(a.probes)}
    (a.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
