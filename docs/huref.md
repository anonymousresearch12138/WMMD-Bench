# HuRef benchmark adapter

The public adapter supplies model/reference loading, tokenizer-specific frozen
top4096 mapping, GQA/fused projection mapping and the ordered six invariant terms.
Native pooling and normalization come from the pinned official checkout, via
[upstream setup](datasets.md); no GAN/encoder training is required.

```bash
python scripts/evaluate.py huref --model-path MODEL --lineage llama --reference REFERENCE_NPY --token-manifest FROZEN_TOKEN_MANIFEST --corpus FROZEN_JSONL --corpus-manifest CORPUS_MANIFEST --output NEW_RESULT_JSON
python scripts/evaluate.py huref --feature CANDIDATE_NPY --reference REFERENCE_NPY --output NEW_RESULT_JSON
```

Reference is the original 512-d float32 feature, SHA256
`ed43610236f89d8f0e367c8f9073b409327e7045d2184e12fdd509ddd2a9e04c`.
`--reference-sha256` permits explicitly binding a separately supplied reference;
the frozen paper threshold is valid only for its original calibration/reference.
Candidate features are also finite 512-d NumPy arrays (`allow_pickle=False`).

Corpus manifest contains `source_sha256` and `joined_text_sha256`. The first 10,000
accepted JSON field records are loaded in the historical field order without text
mutation. Token manifest contains `K:4096`, `invalid_count:0`, `corpus_sha256`, and
4096 ordered `rows` with `token_id`; the adapter rebuilds and verifies the candidate
tokenizer's ordering. Counting uses the original character filter, frequency descending
and token ID descending tie break, excluding exactly `<unk>`, `<s>`, `</s>`.
For Qwen use `--lineage qwen` and its own frozen mapping, never Llama token IDs.

Canonical Llama has 28 layers and Qwen 36. The final two layers supply WqWk,
WvWo and WuWd in that order. FP16 loaded weights are promoted to float32 before
the preserved left-associated products; GQA rows use the historical repeat mapping.
Native pooling reduces the 6x4096x4096 tensor to a globally standardized 512-vector.
ICS is 100 times cosine of globally standardized flattened vectors, positive strictly
above 4.119894027709961. Qwen was included in calibration, not held out.
JSON retains score, frozen threshold, order, mapping and extraction provenance.
Default disk reserve is 100 GiB; the historical 2 GiB continuation option is explicit.

Use [reproduction](reproduction.md) for training and utility, and
[assets](datasets.md) for the frozen input contract. Output paths must be new.
