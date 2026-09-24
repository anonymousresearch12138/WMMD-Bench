# Models, data and external assets

Prepare local snapshots under their owners' licenses. No model, dataset, key or
weight download happens when inspecting the release or reproducing paper figures.

| Model | Pinned revision |
|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| `Qwen/Qwen2.5-3B-Instruct` | `8f4992eda43eea7c770690ddc0de8f732da246f5` |
| `meta-llama/Llama-2-7b-chat-hf` | `f5db02db724555f92da89c216ac04704f23d4590` |

Obtain canonical model/tokenizer snapshots from their official owners or a verified
transport mirror preserving these identities. A different revision is a different
experiment. The selected watermarked Teachers and passive fingerprint packages
are additional benchmark assets; the clean public models cannot replace them.

## Frozen training records

Supply exactly 20,000 JSONL records in their original order. Each contains string
`instruction`, string `input` (possibly empty), and `teacher_raw_answer` for direct
and Bc supervision. Transformed paired records additionally contain
`paraphrased_answer`, retaining the same instruction/input and sample identity.
The response-only mask and chat serialization are implemented in
`wmmd_bench/training/hard_labels.py` and `transformed.py`.

Training checks SHA256 against the explicit `--dataset-sha256`; Bc additionally
checks the method-specific identity in `configs/training/bc.json`. PN-FP Bc and
trajectory use a reconstructed parent (99/1,024 final trajectory matches), separate
from the historical main Ba result (98/1,024). SCW also has a distinct reconstructed
follow-up parent. Do not substitute these parents for the historical Table 1 Ba data.

Response transformation uses frozen Qwen2.5-3B-Instruct with seed 42, temperature
0.7 and top-p 0.9. Preserve atomic answers and failed rewrite fallbacks, sample IDs,
and the recorded method-specific prompt. Not every answer changes. Final processing
counts and exposure statistics are in `results/transformation_and_exposure.json`.
This release consumes the frozen transformed files; it does not claim that a newly
generated paraphrase file is byte-identical to them.

## Detector inputs

The [method matrix](methods.md) lists counts and references. The five detailed
adapter guides define JSON/JSONL schemas and checksum rules. Never regenerate keys
against the paper thresholds. ZeroPrint requires an ordered JSON list containing
two original prompts followed by four perturbations for the first prompt and four
for the second. Retain the frozen `all-mpnet-base-v2` snapshot, perturbations,
reference fingerprint and their checksum manifest; do not replace its embedding model.

SCW's French JSONL and manifest must agree on 1,000 examples and the frozen identity.
iSeal's text dataset, shuffle order, index ranges and secret-key hash are specified
by the A6 config. Utility needs cached ARC-Challenge and TruthfulQA MC2 datasets in
the evaluator's recorded schema, including their original splits and harness version.

There is no confirmed public location for the selected protected Teachers, complete
registered-key packages, reference tensors and frozen generated 20k files.
They must be supplied separately for exact replay. This is an asset availability
limitation, not an implicit requirement for private source code.

## Official code dependencies

No full upstream checkout is bundled. [Licenses](licenses.md) gives all ten pinned
official repositories. For LLMPrint, HuRef and AWM, obtain the permitted official
checkout at `$WMMD_UPSTREAM_ROOT/llmprint`, `/huref`, `/awm`, respectively, then run:

```bash
python -m wmmd_bench.utils.upstream llmprint
python -m wmmd_bench.utils.upstream huref
python -m wmmd_bench.utils.upstream awm
```

The validator checks revision and required-file hashes without importing models.
`configs/evaluation/upstream.json` is the runtime contract. Unknown-license native
source must be obtained under applicable rights; no redistribution grant is implied.
CTCC and EverTracer detector adapters need no native checkout when frozen queries
and neighborhoods are supplied. SCW needs the pinned official `robust_fp` package.

Path bindings in YAML configs are explicit `${...}` placeholders. Resolve them with
`scripts/resolve_config.py` into a separate local file and inspect it before use.
Change storage paths only; preserve the scientific settings and asset hashes.
