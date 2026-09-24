# Reproduction protocol

The Student starts fresh from the pinned clean snapshot for every condition.
Main 3B training: seed/data seed 42, 20k records, three epochs, batch 8,
gradient accumulation 1, 7,500 steps, BF16 full-parameter tuning, max length 1,024,
AdamW, LR 1e-5, zero weight decay, cosine schedule and 225 warmup steps.
Only assistant-response positions contribute to loss.

| Condition | Fresh Student | Supervision |
|---|---|---|
| Ba | Llama 3B | Frozen direct Teacher answer |
| Bb | Llama 3B | Frozen transformed answer |
| Bc | Llama 3B | Direct answer plus full-vocabulary Teacher logits |
| XBa | Qwen 3B | Frozen direct Teacher answer |
| XBb | Qwen 3B | Frozen transformed answer |

Use `scripts/train.py --method METHOD --condition CONDITION` with the local model,
dataset, dataset SHA256 and new output directory. `--validate-only` performs only
path, hash, record-count and schema checks. It does not prove tokenizer or GPU
compatibility. Bb/XBb expect the `paraphrased_answer` field from the frozen paired
file. They do not perform response transformation implicitly.

For Bc also supply `--teacher-path` and `--teacher-tokenizer`. The Teacher is frozen;
the exact historical loss is `0.5 CE + 0.5 * T^2 KL(Teacher || Student)`, T=2, over
causally shifted response tokens. The custom backward preserves the historical
chunked full-vocabulary computation. CTCC loads the original PEFT Teacher adapter
without merging; the Student remains a fresh full-parameter model. Do not substitute
the PN-FP reconstructed-parent data for the historical main Ba comparison.

The five passive detectors evaluate the **same** Student in a condition. Train it
once using `--method passive_shared`, then run LLMPrint/REEF/HuRef/AWM/ZeroPrint.

## Trajectories

Active Ba supports `--trajectory`: save steps 0, 25, 50, 100, 250, 500, 1000,
2000, 4000, 6000 and 7500 in one uninterrupted training run. Step 0 is the clean
Student; Teacher is a separate reference. Evaluate the saved checkpoints with the
same frozen detector after training. This release's checkpoint-saving orchestration
has been CPU reviewed, not rerun on a GPU; it is not a claim to replay historical
wall-clock timings or detector invocation scheduling. Supplied results retain the
original 55 observations and their original loss-observation steps.

## Ownership and utility

Load `final_model` in a fresh evaluation process. Commands and applicability are in
[methods](methods.md). Utility evaluates ARC-Challenge normalized accuracy and
TruthfulQA MC2 with the harness task defaults and chat template:

```bash
python scripts/evaluate.py utility --model-path MODEL --output NEW_JSON
```

Use the recorded evaluator environment for each comparison. Active 3B/cross-lineage
uses lm-eval 0.4.9.1; some historical PN-FP contexts use 0.4.3. Compare each Student
against its matched clean control, not an unrelated cached baseline.

## Representative 7B extension

PN-FP and AWM use Llama-2-7b-chat-hf, seed 42, 20k records and the same horizon,
but AdamW8bit 0.50.0; this is not a controlled scaling ablation. The retained workers
are `wmmd_bench.training.scale7b_direct_paraphrase` and `scale7b_logit`.

```bash
python -m wmmd_bench.training.scale7b_direct_paraphrase train --base BASE_7B --dataset DIRECT_JSONL --output NEW_DIR
python -m wmmd_bench.training.scale7b_logit cache --base BASE_7B --teacher TEACHER_7B --dataset DIRECT_JSONL --output CACHE_DIR
python -m wmmd_bench.training.scale7b_logit train-kd --base BASE_7B --dataset DIRECT_JSONL --cache CACHE_DIR --output NEW_DIR
```

For transformed hard targets, use the frozen 7B Student-ready file with its selected
answer in `teacher_raw_answer`. The two workers retain the recorded cache formats;
do not interchange caches. Full-vocabulary BF16 logits require substantial disk.
Executed settings and model identity are in `configs/training/7b/`. AWM evaluation
uses `python scripts/evaluate.py awm ... --scale 7b`. PN-FP uses the frozen 7B
one-token response signature (key length 16). No LLMPrint 7B result or cross-lineage 7B result
is claimed. Exact replay still requires the external selected Teachers and data.
