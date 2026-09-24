# Licenses and upstream revisions

The root MIT grant covers WMMD-Bench-authored portions only. Modified third-party
integrations retain upstream terms and attribution. No full upstream checkout is bundled.
The table below records the inspected official revisions and license observations.

| Method | Pinned upstream | License | Distributed scope |
|---|---|---|---|
| pnfp | [fdceaba14bd3e89340916a6a40e27c945d48460e](https://github.com/SewoongLab/scalable-fingerprinting-of-llms/tree/fdceaba14bd3e89340916a6a40e27c945d48460e) | MIT | Modified licensed integration; see NOTICE |
| evertracer | [70b402f7b7456c6d94e1fae2de554d77dd6cd921](https://github.com/Xuzhenhua55/EverTracer/tree/70b402f7b7456c6d94e1fae2de554d77dd6cd921) | No explicit grant found at inspected revision | Benchmark-authored adapter; native upstream code obtained separately |
| ctcc | [8db93218260bed31b8f18acc9c6ac3e1955d3a42](https://github.com/Xuzhenhua55/CTCC/tree/8db93218260bed31b8f18acc9c6ac3e1955d3a42) | No explicit grant found at inspected revision | Benchmark-authored adapter; native upstream code obtained separately |
| iseal | [7e382321eef4355002acd93120d888dc9b45a8bd](https://github.com/IntelliSys-Lab/iSeal/tree/7e382321eef4355002acd93120d888dc9b45a8bd) | MIT | Modified licensed integration; see NOTICE |
| scw | [15bc1929569357130f2dbc0b09f91bbf4f4bd947](https://github.com/eth-sri/robust-llm-fingerprints/tree/15bc1929569357130f2dbc0b09f91bbf4f4bd947) | Responsible AI SOURCE CODE License v1.1 | Modified licensed integration; see NOTICE |
| llmprint | [3e577f98b2bb64780ec2995b074c5aeec9b017e1](https://github.com/hifi-hyp/ACL-LLMPrint/tree/3e577f98b2bb64780ec2995b074c5aeec9b017e1) | No explicit grant found at inspected revision | Benchmark-authored adapter; native upstream code obtained separately |
| reef | [48329f6f3695a8aea33975832159e7ce44ad73f9](https://github.com/AI45Lab/REEF/tree/48329f6f3695a8aea33975832159e7ce44ad73f9) | Apache-2.0 (README declaration; no LICENSE file in pinned tree) | Modified licensed integration; see NOTICE |
| huref | [9c34548a6f6c1e78780e1fd07de56c7a3357f6ef](https://github.com/LUMIA-Group/HuRef/tree/9c34548a6f6c1e78780e1fd07de56c7a3357f6ef) | No explicit grant found at inspected revision | Benchmark-authored adapter; native upstream code obtained separately |
| awm | [bc20ff8e63cec57f5da422ae065686ced275e76d](https://github.com/LUMIA-Group/AWM/tree/bc20ff8e63cec57f5da422ae065686ced275e76d) | No explicit grant found at inspected revision | Benchmark-authored adapter; native upstream code obtained separately |
| zeroprint | [16a02aa4cfd5693ecfa757e9d2832b7e9babada0](https://github.com/shaoshuo-ss/ZeroPrint/tree/16a02aa4cfd5693ecfa757e9d2832b7e9babada0) | MIT | Modified licensed integration; see NOTICE |

For the five projects with no explicit grant at the pinned revision, no grant to
redistribute or modify their source is inferred. Obtain their code under applicable
rights separately. This release includes benchmark-authored loading, mapping,
evaluation, aggregation and serialization code, plus validated native component calls.
It does not bundle unknown-license upstream implementations or unverified patches.
Mixed historical files are not copied whole; only the required benchmark adapters are retained.

When using those components, place the official pinned checkout at
`${WMMD_UPSTREAM_ROOT}/METHOD` and run `python -m wmmd_bench.utils.upstream METHOD`.
The validator checks Git revision and required-file SHA256 without importing models.
The exact URLs, revisions and files are in
[upstream dependencies](../configs/evaluation/upstream.json).

PN-FP, iSeal and ZeroPrint integrations retain MIT notices; SCW retains its
Responsible AI SOURCE CODE License v1.1 and Exhibit A restrictions. REEF declares
Apache-2.0 in its pinned README; that tree has no LICENSE file. The standard Apache
text and attribution are retained. These observations do not license external assets.

## Distributed scope

The training formatter/objective, shared orchestration, five benchmark adapters
(CTCC, EverTracer, LLMPrint, HuRef, AWM), plotting and documentation are first-party.
PN-FP, iSeal and ZeroPrint integrations retain MIT attribution; SCW helper/integration
code retains its source license; REEF extraction/scoring retains Apache-2.0.
Files in `wmmd_bench/methods/` with these licensed method prefixes, and the retained
7B PN-FP training integrations, are treated conservatively as adapted portions.
No license claim is made over separately supplied models, datasets or native code.
