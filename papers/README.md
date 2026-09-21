# Research source index

Authored 2026-09-06. The previous manuscript prose in this directory was retired
under the owner's model-provenance policy. This replacement is a source inventory,
not a paper, independent replication, novelty review, or restored historical result.

## Available source

The following repository paths identify implementations and experimental tools.
Presence establishes where to inspect code, not that inputs, external model
weights, dependencies, or a runnable environment are available. Read each entry
point before execution; this index does not authorize downloads or compute spend.

| Collection | Repository source |
|---|---|
| Numerical reproductions | `research/papers/run_ditherdmd.py` |
| Numerical reproductions | `research/papers/run_hurc.py` |
| Numerical reproductions | `research/papers/run_pseudospectral.py` |
| Numerical reproductions | `research/papers/run_qpregret.py` |
| Numerical reproductions | `research/papers/run_speccert.py` |
| Numerical reproductions | `research/papers/run_srcvar.py` |
| Retained model experiments | `research/experiments/expllm/exp_kv.py` |
| Retained model experiments | `research/experiments/expllm/exp_kv2.py` |
| Retained model experiments | `research/experiments/expllm/exp_kv_figure.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_partA.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_partA_scale.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_partB.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_partB2.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_partC.py` |
| Retained model experiments | `research/experiments/expllm/exp_llm_figure.py` |
| Retained model experiments | `research/experiments/expllm/qwen2_forward.py` |
| Protocol and evaluation | `research/experiments/accs/spec.py` |
| Protocol and evaluation | `research/experiments/accs/verifier.py` |
| Protocol and evaluation | `research/experiments/accs/reference.py` |
| Protocol and evaluation | `research/experiments/accs/exhaustive.py` |
| Protocol and evaluation | `research/experiments/accs/coverage.py` |
| Protocol and evaluation | `research/experiments/accs/benchmarks/evaluation.py` |
| Protocol and evaluation | `research/experiments/accs/benchmarks/ablation.py` |

## Evidence and publication

Retained JSON records and code must be interpreted with their original input,
seed, hardware, and version metadata. Missing evidence stays unmeasured. A new
experiment must preserve failure cases and state its comparison before execution.

The private source tree and public package are distinct distributions. This
index does not promise that every listed research script is part of a public
wheel or mirror. Public export requires its existing allowlist and review gates;
retirement of private prose does not itself publish or retract a public artifact.
