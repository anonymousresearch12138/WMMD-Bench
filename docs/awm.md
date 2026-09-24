# AWM benchmark adapter

The public adapter loads model weights, selects Wq/Wk, maps dimensions/layers with
the historical LAP orchestration, calls official unbiased linear CKA and aggregates
native ownership scores. Install the pinned `awm` checkout through
[upstream setup](datasets.md).

```bash
python scripts/evaluate.py awm --model-path STUDENT --reference-model BASE_3B --scale 3b --output NEW_RESULT_JSON
python scripts/evaluate.py awm --model-path STUDENT_7B --reference-model BASE_7B --scale 7b --output NEW_RESULT_JSON
python scripts/evaluate.py awm --saved-alignment ALIGNMENT_JSON --scale 3b --output NEW_RESULT_JSON
```

Reference is the clean canonical Llama Base: 28 layers for 3B or 32 for 7B.
Weights load as FP16 on CPU and selected tensors promote to float32, matching the
historical worker. Vocabulary overlap is sorted by token string. Dimension LAP
minimizes 1-absolute cosine of normalized embedding columns; permutation/sign
mapping, zero-sign handling and target ordering are preserved. This cosine is only
the alignment cost, never the ownership metric. Wq/Wk use no GQA row expansion.

Equal layer counts preserve identity layer order. Otherwise layer LAP minimizes
negative mean Wq/Wk unbiased CKA. Native CKA receives transposed aligned weights,
linear kernel and `unbiased=True`; per-layer means are taken separately for Wq and
Wk, then averaged. Frozen strict thresholds are 0.0017953364917795106 (3B) and
0.007566815861277831 (7B); ties are negative. No calibration is rerun.

Saved alignment is a `layer_alignment` object (or that object directly) containing
`per_layer`: unique `reference_layer`, `candidate_layer`, `Wq_weights`, `Wk_weights`.
Replay recomputes means without loading models/upstream. Full mode emits dimension
and layer alignment metadata plus score/threshold/decision. Qwen was in the 3B
calibration panel, not held out. Default device is CUDA; `--device cpu` exists for
small CPU regression and does not validate the formal model execution stack.

Use [reproduction](reproduction.md) for training and utility, and
[assets](datasets.md) for the frozen input contract. Output paths must be new.
