"""Dependency-free throughput arithmetic for measured concurrency sweeps.

A sweep row is ``(workers, wall_seconds)`` for ``workers`` independent units
completed in that wall time.  Keeping the conversion and knee rule here lets
callers bring measurements from different workloads without importing the
repository's CI registry or copying its arithmetic.

The functions deliberately preserve the input order.  A sweep is evidence from
successive concurrency levels, not an unordered collection that this module may
silently sort or repair.
"""
from __future__ import annotations


#: A marginal worker earns its place while it returns at least this share of
#: the first worker's measured rate.
DEFAULT_MARGINAL_FLOOR = 0.5


def independent_sweep_throughput(
        sweep) -> tuple[tuple[int, float], ...]:
    """Return ``(workers, units_per_second)`` for an independent-unit sweep."""
    rows = tuple(sweep)
    return tuple((workers, workers / wall) for workers, wall in rows)


def throughput_knee(
        sweep, *, floor: float = DEFAULT_MARGINAL_FLOOR) -> int:
    """Return the last worker count whose marginal worker paid for itself.

    ``floor`` is measured against the first worker's rate.  The first failing
    step ends the admissible range; later rows cannot make an earlier
    contention loss disappear.
    """
    measured = independent_sweep_throughput(sweep)
    base = measured[0][1]
    knee = measured[0][0]
    previous = measured[0]
    for workers, rate in measured[1:]:
        added = workers - previous[0]
        if added and (rate - previous[1]) / added >= floor * base:
            knee = workers
        else:
            break
        previous = (workers, rate)
    return knee


__all__ = [
    "DEFAULT_MARGINAL_FLOOR",
    "independent_sweep_throughput",
    "throughput_knee",
]
