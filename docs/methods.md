# Methods and native measurements

All Student conditions use `scripts/train.py`. Active methods require their
selected protected Teacher; passive methods use the canonical clean Llama reference
and a shared response dataset/Student. Ba/XBa use `teacher_raw_answer`, Bb/XBb use
`paraphrased_answer`, and Bc adds the frozen Teacher's response-position logits.

| Method | Protected source / required frozen assets | Implementation under `wmmd_bench/methods/` | Native Table 1 measurement | Qwen applicability |
|---|---|---|---|---|
| PN-FP | Selected A2 Teacher; 1,024 keys; canonical Llama secret tokenizer | `pnfp.py`, `pnfp_cross.py` | Exact matches / 1,024 | Frozen Llama strings, full encoded target equality; canonical |
| EverTracer | Protected Teacher; 100 member + 100 nonmember frozen neighborhoods, each with ten variants | `evertracer/` | Member-oriented ROC-AUC; supplementary TPR at empirical FPR <= .05 | Canonical N/A; exploratory AUC stays separate |
| CTCC | Protected Teacher; original frozen PEFT adapter for Bc; annotated 95 trigger, 100 suppression, 105 normal queries | `ctcc/` | Exact stripped `IAMALIVE` trigger matches / 95, negative controls / 205 | Operationally applicable |
| iSeal | Selected A6 Teacher and canonical Base; secret cipher key; registered 200 + held-out 100 text indices | `iseal.py`, `iseal_cipher.py` | Registered sentence-BLEU >= 50 successes / 200 | Level-3 canonical N/A |
| SCW | Selected Teacher; 1,000 frozen French queries; native watermark config and matching tokenizer | `scw.py`, `scw_generate.py` | Native p-value; positive when p < .001 | Audited tokenizer/alignment protocol; canonical |
| LLMPrint | Clean reference; 200 registered fingerprint artifacts and reference probability sequence | `llmprint/` | Primary paper-style bit accuracy; >= frozen .7150049776126003 | Canonical; supplementary released rule is distinct |
| REEF | Clean reference representation, 200 frozen probes | `reef.py`, `reef_core.py` | Centered linear CKA; strictly > .4546738923165847 | Last raw Qwen decoder block, distinct width allowed |
| HuRef | Clean reference feature, frozen 10k corpus and tokenizer-specific 4,096-token manifests | `huref/` | Native ICS; strictly > 4.119894027709961 | Audited GQA/token mapping; canonical |
| AWM | Clean reference weights/tokenizer; original alignment and calibration | `awm/` | Mean Wq/Wk unbiased linear CKA; strictly > frozen threshold | Native dimension/layer assignment; canonical |
| ZeroPrint | Clean reference fingerprint; 2 HumanEval prompts + 8 frozen perturbations; frozen MPNet snapshot | `zeroprint.py`, `zeroprint_core.py` | Rescaled Pearson; strictly > .6793505996465683 | Same frozen query/representation protocol |

PN-FP, CTCC and iSeal counts and EverTracer AUC are measurements, not converted
into a common binary ownership label. Cross-lineage N/A is not a zero score.
Clean Qwen participated in the frozen REEF/AWM/ZeroPrint negative calibration;
do not interpret its comparison as an independent held-out specificity test.

## Detector commands

Detailed frozen schemas: [EverTracer](evertracer.md), [CTCC](ctcc.md),
[LLMPrint](llmprint.md), [HuRef](huref.md), [AWM](awm.md).

```bash
python scripts/evaluate.py pnfp --model-path MODEL --fingerprints KEYS_JSON --output RESULT_JSON --label student --use-chat-template --training-response-length 1
python scripts/evaluate.py pnfp_cross --model MODEL --fingerprints KEYS_JSON --fingerprints-sha256 SHA --secret-tokenizer LLAMA_BASE --output NEW_DIRECTORY
python scripts/evaluate.py iseal --config RESOLVED_A6_YAML --teacher MODEL --dataset-cache DATASET_CACHE --output RESULT_JSON
python scripts/evaluate.py scw_generate --model MODEL --label student --eval-jsonl FRENCH_JSONL --eval-manifest FRENCH_MANIFEST --output GENERATIONS_JSONL
python scripts/evaluate.py scw --protocol-config RESOLVED_SCW_YAML --official-config NATIVE_YAML --generations GENERATIONS_JSONL --model MODEL --output RESULT_JSON
python scripts/evaluate.py reef --model-path MODEL --reference REPRESENTATION_NPY --probes PROBES_JSON --output-dir NEW_DIRECTORY
python scripts/evaluate.py zeroprint --model-path MODEL --mpnet MPNET_SNAPSHOT --prompts TEN_PROMPTS_JSON --prompts-sha256 SHA --reference REFERENCE_NPY --reference-sha256 SHA --output-dir NEW_DIRECTORY
```

iSeal requires `ISEAL_SECRET_KEY_HEX`; the supplied hash must match the A6 config.
Use the selected Teacher's tokenizer/Base, not a new cipher or newly selected text.
For SCW, explicitly bind the protocol model path to the evaluated checkpoint and
preserve all other settings. Install `robust_fp` from the pinned official checkout.
For REEF Qwen evaluation add `--cross-lineage`; reference is always the clean Llama
representation. PN-FP 7B retains key length 16 and one target response token;
use `--training-response-length 1`, the frozen 7B keys and the matching chat template.

ZeroPrint's internal generation seeds start at 1000, with 20 samples per prompt.
These native detector sampling streams are not Student training seeds. HuRef's
standardization and AWM's layer averaging are native feature operations, not
aggregation over independent Student runs.
