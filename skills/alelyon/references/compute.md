# The compute DAG — uncertainty propagation and variance attribution

Load this when a result depends on several uncertain inputs and the point estimate alone
would be misleading.

```python
from alelyon.runtime.vector.compute import (
    ComputationGraph, Constant, Normal, TruncatedNormal, Empirical,
)
```

## Build, evaluate, propagate

Construction order is free — node existence is validated lazily at evaluation time, so a
node may reference dependencies added later.

```python
g = (ComputationGraph()
     .add_input("units", Normal(1000, 50))
     .add_input("price", Normal(19.99, 0.5))
     .add_input("cost",  Normal(11.00, 1.20))
     .add("revenue", lambda i: i["units"] * i["price"], ["units", "price"])
     .add("margin",  lambda i: i["revenue"] - i["units"] * i["cost"],
          ["revenue", "units", "cost"]))

g.evaluate()                              # point value of every node
r = g.propagate(n_samples=20_000, seed=11)
```

`fn(inputs)` receives a dict mapping each dependency name to its value — a scalar under
`evaluate`, an `(n,)` array under `propagate`. Write node functions to work with both;
plain arithmetic and NumPy ufuncs already do.

## Reading the result

`GraphResult` has four fields: `results`, `sensitivities`, `samples`, `n_samples`.

```python
m = r.results["margin"]      # NodeResult: name point mean std q05 q50 q95
r.sensitivities["margin"]    # variance share per source node
```

Measured output of the graph above:

```
margin: point 8990  mean 8995  sd 1380  q05 6776  q95 11293
variance share: {'cost': 0.755, 'price': 0.134, 'units': 0.111}
```

## Why this matters for how you report

The point estimate is `8990`. The 5th–95th interval is `6776 … 11293`. Reporting the
point alone implies a precision the inputs do not support — that is the
observed-versus-declared failure in numeric clothing.

**Attribution is the actionable part.** Cost uncertainty carries 75.5% of the variance in
margin; tightening the price estimate would move almost nothing. When you present a
propagated result, lead with the interval and the dominant contributor, not the mean.

## Determinism

`seed=` builds the RNG; pass an existing `MonteCarloSimulator` via `simulator=` to share
one. Always pass a seed in anything you will report or test — an unseeded run gives a
different interval each time, and a number that changes when rerun cannot be checked by
whoever reads it.

`keep=` returns raw `(n,)` sample arrays for named nodes (default: none, to avoid
carrying large arrays). `attribute=` chooses which targets get variance decomposition
(default: every sink).

## Honest limits

- These are **Monte-Carlo** estimates. The interval has its own sampling error; raising
  `n_samples` narrows it. Do not quote propagated quantiles to more precision than the
  sample size supports.
- `sensitivities` is a **variance share under the sampled joint distribution**, not a
  causal claim and not a derivative. It answers "where does the spread come from",
  not "what happens if I change this".
- Input distributions are **declared by you**. `Normal(11.00, 1.20)` is an assertion
  about cost uncertainty. The propagation is exactly as good as that assertion, and the
  library cannot check it.
