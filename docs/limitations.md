# Scope and limitations

This is a code-and-compact-results release, not a full experimental archive.
Exact replay requires selected protected Teachers, registered keys, frozen generated
data and reference features that are not bundled and have no confirmed public
download location. Replacing them creates a new experiment.

The release includes all ten detector integrations and Student training code.
It does not claim to reproduce protected Teacher acquisition from scratch solely
from this checkout. Official native dependencies with no confirmed redistribution
grant remain external. There is no private helper-module dependency.

Main 3B and 7B Student training use seed 42 only. Repeated-run uncertainty was not
measured. Trajectory checkpoints are dependent observations within one continuous
run. PN-FP/SCW reconstructed follow-ups are distinct from historical Ba endpoints.
EverTracer cross-tokenizer and iSeal cross-architecture canonical results remain N/A.

CPU release checks establish packaging, numerical consistency and selected loss/
detector mathematics. They do not establish equivalence of every GPU execution,
environment, tokenizer/date-dependent template, generated answer or utility cache.
Historical environments differ by stage and are not one fully locked environment.
The new explicit-path training orchestration has not been scientifically rerun.

The representative 7B extension changes optimizer implementation; it is not an
isolated model-size ablation. The interrupted LLMPrint 7B construction is excluded
from reported results. Native detector sampling seeds, feature standardization,
layer averaging and within-query statistics remain distinct from training repeats.
