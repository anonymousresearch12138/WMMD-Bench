# WMMD-Bench

**Large Language Model Watermarks under Model Distillation**

WMMD-Bench evaluates five active watermarks and five passive fingerprints after
direct-response, transformed-response and soft-logit distillation. It retains each
method's native detector and compares ownership measurements with clean-model
controls and utility evaluation.

**All main 3B Student training uses a single seed, 42.** Endpoint results denote
detector measurements on the final checkpoint of that one training run. The 55
trajectory observations come from five continuous runs, with eleven checkpoints
per active method. The representative 7B extension also uses seed 42.

## Start here

| Task | Entry point |
|---|---|
| Understand all ten methods and their assets | [Method matrix](docs/methods.md) |
| Obtain models, frozen data, keys and official dependencies | [Data and models](docs/datasets.md) |
| Train one fresh Student | `python scripts/train.py --help` |
| Run one ownership detector or utility evaluator | `python scripts/evaluate.py --help` |
| Reproduce numerical tables and figures without a GPU | `python scripts/reproduce_paper_results.py` |
| Check the release and numerical schemas | `python scripts/check_release.py` |

## Reproduce the paper presentation on CPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements/plotting.txt
python scripts/reproduce_paper_results.py
python scripts/check_release.py
```

Seven small result files in `results/` supply Table 1, the numerical appendix tables,
and the five numerical figures. Outputs go to `paper/tables/` and `paper/figures/`.
This renders recorded measurements; it does not rerun model experiments.
See the [result schemas and paper mapping](docs/results.md).

With NumPy and a CPU PyTorch environment installed, `python tests/test_cpu.py`
checks the response-loss gradient and detector boundary cases without loading models.

## Train and evaluate

Scientific execution targets Linux/CUDA with separately prepared local assets.
Use the stage-specific environments in [dependencies](docs/dependencies.md).

```bash
python scripts/train.py --method pnfp --condition Ba \
  --model-path /data/base-3b --dataset /data/pnfp-ba.jsonl \
  --dataset-sha256 FROZEN_SHA256 --output-dir /data/runs/pnfp-ba
python scripts/evaluate.py pnfp --model-path /data/runs/pnfp-ba/final_model \
  --fingerprints /data/pnfp-keys.json --output /data/pnfp-ba.json \
  --label student --use-chat-template --training-response-length 1
```

The main models are Llama-3.2-3B-Instruct and, for cross-lineage Students,
Qwen2.5-3B-Instruct. Ba/Bb/Bc use fresh Llama Students; XBa/XBb use fresh Qwen
Students. All use 20,000 frozen records, three epochs, batch eight and 7,500 updates.
The five passive methods share a Student within each condition; they do not require
five duplicated training runs. See [reproduction](docs/reproduction.md).

Selected protected Teachers, frozen generated data and complete fingerprint
packages are **not bundled and do not yet have a confirmed public download
location**. They are required for exact replay. A new generated package defines a
new experiment. This limitation is documented in [limitations](docs/limitations.md).
There is no private Python implementation extension.

WMMD-Bench-authored code uses MIT. Adapted integrations retain their original
licenses; some native components must be obtained from pinned official repositories.
See [licenses](docs/licenses.md) and [NOTICE](NOTICE).
