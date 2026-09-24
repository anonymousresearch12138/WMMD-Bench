# CTCC benchmark adapter

The public operational detector performs model inference on frozen queries, then
case-sensitive `strip(response) == "IAMALIVE"`. It requires no upstream runtime for
evaluation; Teacher construction remains a separate upstream/assets task.

```bash
python scripts/evaluate.py ctcc --model-path MODEL --queries QUERY_FILE --output NEW_RESULT_JSON
python scripts/evaluate.py ctcc --saved-generations SAVED_JSON --output NEW_RESULT_JSON
```

`QUERY_FILE` is a JSON list of 300 entries: `category` is `trigger` (95),
`suppression` (100) or `normal` (105); `record` contains `instruction`, `input`,
`output`, and optional `history` user/assistant pairs. The expected output is removed
before prompting. Native chat serialization, EOS left padding, BF16/SDPA, batch 8,
max input length 2048 and greedy 96-token decoding preserve the historical worker.
Optionally bind the entire query file with `--queries-sha256 SHA256`.

Saved input is either a list or an object with `raw_generations`, each containing
`category`, `raw_generation`, and optional `generation_error: null`. The adapter
recomputes matches; a generation error fails evaluation rather than becoming a negative.
JSON reports all group counts/rates and false activations over 205 negatives. The
benchmark endpoint is trigger activation count/95, not a newly invented binary
model-level threshold; `ownership_decision` is null for that reason.

Use [reproduction](reproduction.md) for training and utility, and
[assets](datasets.md) for the frozen input contract. Output paths must be new.
