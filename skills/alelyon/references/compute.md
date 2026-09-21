# Compute graph

Use the Vector computation graph for a calculation whose inputs carry declared
uncertainty. It evaluates dependency nodes and can propagate sampled inputs
through the same functions.

## Example

```python
from alelyon.runtime.vector.compute import ComputationGraph, Normal

graph = (
    ComputationGraph()
    .add_input("volume", Normal(100, 5))
    .add_input("unit_margin", Normal(8, 1))
    .add("margin", lambda values: values["volume"] * values["unit_margin"],
         ["volume", "unit_margin"])
)
points = graph.evaluate()
result = graph.propagate(n_samples=4000, seed=17)
margin = result.results["margin"]
print(margin.point, margin.mean, margin.std, margin.q05, margin.q95)
print(result.sensitivities["margin"])
```

The distribution parameters are example declarations, not measured business data.
Run the example to obtain output for your installed build.

## Function contract

Node functions receive their named dependencies in a mapping. During point
evaluation values are scalars; during propagation they are sample arrays. Use
elementwise functions that support both forms. Do not reduce across the sample
axis inside a node.

Dependencies can be registered before their nodes exist, but evaluation checks
the graph. Inspect failure results instead of hiding missing dependencies or
invalid inputs behind a default output.

`GraphResult` carries `results`, `sensitivities`, `samples` and `n_samples`.
Each `NodeResult` includes the point estimate and sampled summary statistics.
Use `keep=` to retain selected sample arrays and `attribute=` to choose
attribution targets. Retaining arrays has a memory cost.

## Interpretation

Report the seed, sample count, declared distributions and sampled interval.
A larger sample can reduce uncertainty in estimated quantiles; it does not
eliminate uncertainty in the inputs or guarantee a narrower outcome interval.

Variance attribution describes the sampled model. It is not a causal effect or
proof that changing one input in the real world will have the same effect.
Input distributions remain the caller's responsibility.

Source-checkout reference: `alelyon/runtime/vector/compute/`.
Focused checks: `tests/vector/test_compute_graph.py`.
