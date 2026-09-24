# LLMPrint benchmark adapter

The public adapter extracts the next-token probability/logit sequence from a model
and frozen fingerprints, then calls the pinned official scoring helpers. Obtain
`llmprint` via [upstream setup](datasets.md). GCG construction is
not needed to evaluate a supplied fingerprint package.

```bash
python scripts/evaluate.py llmprint --model-path MODEL --fingerprint-dir FINGERPRINT_DIR --fingerprint-manifest MANIFEST_JSON --reference REFERENCE_SEQUENCE_JSON --output NEW_RESULT_JSON
python scripts/evaluate.py llmprint --sequence CANDIDATE_SEQUENCE_JSON --reference REFERENCE_SEQUENCE_JSON --calibration FROZEN_DUAL_CALIBRATION_JSON --output NEW_RESULT_JSON
```

The fingerprint manifest must have `validated_record_count: 200`,
`integrity_status: "PASS"`, and ordered `artifacts` with `pair_id` and `sha256`.
Files `llmprint-pair-000.json` through `llmprint-pair-199.json` contain `w_plus`,
`w_minus`, and `complete_optimized_prompt`. Reference and candidate must use the same
manifest hash and ordered target words. 500 is the historical GCG construction
budget; the detector sequence has 200 positions.

The recovered probability worker tokenizes without special tokens, uses each target
word's first token, and takes float32 softmax over last-token logits. Default model
dtype is float16; use `--dtype bfloat16` only for the matching recorded stage.
Primary bits use positive logit >= negative logit (ties one), accuracy over all 200;
positive means accuracy >= 0.7150049776126003. Logits preserve the historical paper
bits even when probabilities underflow.

The optional supplementary released rule uses strict probability comparison,
all four probabilities > .001, exact one-sided binomial p < .05, and the supplied
frozen `supplementary.calibrated_min_n`. Calibration must also contain
`primary.final_tau` equal to the frozen primary threshold. It never replaces the
primary endpoint. Output includes separate primary/supplementary results and the
candidate sequence. Qwen belongs to the historical calibration panel, not a held-out panel.

Use [reproduction](reproduction.md) for training and utility, and
[assets](datasets.md) for the frozen input contract. Output paths must be new.
