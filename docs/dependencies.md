# Dependencies
The inspection CLI and table renderer use the Python standard
library. Configuration parsing uses PyYAML. Scientific workers target Linux; some use
`fcntl`, `resource` and POSIX filesystem behavior and are not Windows runtime ports.

| Component | Required libraries | Version evidence / limitations |
|---|---|---|
| 3B distillation and generation | PyTorch, Transformers, Accelerate, datasets, NumPy, PEFT, SentencePiece | `requirements/core-3b.txt`; PyTorch/CUDA stage identity still required |
| PN-FP native / earlier workers | Above plus DeepSpeed | `requirements/pnfp-legacy.txt`; external pinned PN-FP code may add dependencies |
| 7B training | PyTorch 2.8.0 CUDA 12.8 build, Transformers 4.44.2, bitsandbytes 0.50.0; native Teacher also DeepSpeed | `requirements/scale-7b.txt` is a partial recorded stack, not a full lock |
| ARC / TruthfulQA utility | lm-eval and its dependencies | 0.4.9.1 in active 3B; legacy PN-FP uses 0.4.3. Do not substitute evaluator versions |
| iSeal | sacrebleu, model libraries and the native adapter dependencies | Exact sacrebleu version is not established in the retained release evidence |
| REEF / HuRef | NumPy, PyTorch, Transformers | Preserve stage dtype, layer extraction and tokenizer behavior |
| AWM | SciPy LAP, model libraries, external `similarity_metrics` | AWM 7B detector recorded Transformers 4.51.3, separate from training 4.44.2 |
| ZeroPrint | NLTK + frozen language resources, model libraries, MPNet snapshot | NLTK version/resources and source-specific requirements still need a complete lock |
| SCW | external `robust_fp` and its pinned dependencies | Upstream requirements and license must accompany any later vendoring |
| LLMPrint | pinned native GCG implementation + model libraries | External requirements must be matched to the recorded source |
| Figures | NumPy 1.26.4, Matplotlib 3.9.1 | `requirements/plotting.txt`; no models used |

Known conflicts: Transformers 4.44.2 / 4.51.3 / 4.55.2, Accelerate 0.32.1 /
1.10.0 / 1.10.1, NumPy 1.26.4 / 2.2.6 / 2.3.2, datasets 2.20.0 / 3.6.0,
PEFT 0.11.1 / 0.17.1, PyYAML 6.0.1 / 6.0.2 and lm-eval 0.4.3 / 0.4.9.1
occur in different recorded stages. They are not one jointly installable lock file.
The cross-lineage environment also recorded SciPy 1.16.2; its exact full environment
needs validation rather than installation from a broad machine snapshot.

Pandas, pydantic, scikit-learn, wandb and psutil are not direct imports of the bundled
scientific workers. They are not included in the minimal lists merely because they
appeared in machine package inventories. Native external projects may require them.
Dependencies of those projects have not been proven removable.

CPU packaging checks were exercised with Python 3.12.14, PyYAML 6.0.2,
Matplotlib 3.9.2 and NumPy 2.3.5. The recorded plotting requirement remains 3.9.1;
the Windows presentation check used 3.9.2. This is not validation of any scientific
environment or GPU execution stack.
Use separate environments; validate them against the corresponding experiment before
scientific execution. Unrecorded versions are unresolved, not implicitly “latest”.

## CPU verification environment

The release uses only standard-library inspection for basic checks. Optional CPU
mathematical tests use PyTorch 2.8.0+cpu, NumPy 2.3.5, SciPy 1.16.2 and safetensors
0.6.2 with the validated native checkouts. No model inference is involved.
LLMPrint evaluation uses `detect_func.py`, not GCG construction. HuRef's selected
native functions need no GAN/encoder dependencies. AWM native CKA imports safetensors.
