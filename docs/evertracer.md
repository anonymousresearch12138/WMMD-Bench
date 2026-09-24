# EverTracer benchmark adapter

The public detector includes frozen-neighborhood loading, sequence probability
inference, calibration, corrected member orientation, AUC and TPR at empirical
FPR <= .05. It needs no upstream attack model at inference when neighborhoods are
already supplied. Constructing those assets may use the pinned official attack code.

```bash
python scripts/evaluate.py evertracer --model-path STUDENT --reference-model REFERENCE --tokenizer-path CLEAN_LLAMA --neighborhoods FROZEN_JSONL --output NEW_RESULT_JSON
python scripts/evaluate.py evertracer --saved-scores SCORES_JSON --output NEW_RESULT_JSON
python scripts/evaluate.py evertracer --model-path QWEN_STUDENT --lineage qwen --output NEW_NA_JSON
```

The frozen JSONL contains 100 `dtr` and 100 `dunseen` rows with `original`,
`position`, `source_index`, and five `pairs`, each with `positive` and `negative`
strings. Use the canonical clean Llama tokenizer and original reference model.
BF16 causal sequence probability is exp(-mean loss), truncation length 128.
Variation is the mean of ten neighbor probabilities minus original probability;
C is suspect variation minus reference variation. Members are positive under -C.

Saved scores are a list or `scores` field, with `subset` and `calibrated_score`.
The output retains the continuous native AUC/TPR, empirical FPR and threshold.
There is no added model-level binary threshold. A reject-all threshold is serialized
as the JSON string `Infinity`. Qwen cross-tokenizer canonical results stay N/A;
exploratory AUC is not promoted into ownership evidence.

Use [reproduction](reproduction.md) for training and utility, and
[assets](datasets.md) for the frozen input contract. Output paths must be new.
