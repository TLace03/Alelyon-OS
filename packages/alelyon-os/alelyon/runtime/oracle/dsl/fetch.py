"""Feeding the certified engine from data you hold.

`execcert.certified_run` takes a `fetcher`, executes a restricted program over
what it returns, and produces a bound on how far storage quantization can have
moved the answer. Without a fetcher there is nothing to certify, and the only
one that shipped read a private market-data store — so the public package could
verify somebody else's envelope and could not produce one of its own. This is
the missing half.

What a fetcher is
-----------------
One method:

    get(kind: str, key: str) -> FetchedSeries

`kind` is `"price"`, `"series"` or `"table"` — whichever the program's
references use — and `key` names the thing. It returns the stored values, a
per-row quantization step, and a count of rows that carry no certificate.

Δ IS A DECLARATION, AND THE ENVELOPE RECORDS IT AS ONE
------------------------------------------------------
This is the load-bearing sentence in this module, and it is the one most likely
to be skipped.

`delta` is your statement about **how your data was stored** — the width of the
interval a stored value could have come from. The certificate then bounds how
far that storage step can have moved the final scalar. It is arithmetic over a
declared premise.

It is **not** a claim that your numbers are right. Declaring `delta=0` on
invented data yields a certificate of width zero over invented data. Nothing in
this module, in `certified_run`, or in the verifier can detect that, and none of
them claim to: the receipt detects **revision of committed inputs**, not
fabrication at capture. A verifier re-deriving your envelope reproduces your
arithmetic under your declaration; it does not audit your source.

So the honest reading of an envelope produced here is: *given that these values
were stored under this law with this step, the answer is x ± w, and anyone can
re-derive that.* Everything before "given" is yours to stand behind.

Capture laws
------------
`law` names the rule that produced the step, because Δ means different things
under different rules and a Δ read without its law is how a fake zero gets back
in (`attest.KNOWN_CAPTURE_LAWS`):

    None                  relative-dither. Δ=0 means the column was all zero.
    "dither-relative/v0"  the same, named explicitly.
    "exact-cents/v0"      whole-cent monetary storage. Δ=0 means the values ARE
                          whole cents, which is a representability claim about
                          every value, and `certified_run` enforces an aggregate
                          2^53 guard because of it.

An unrecognised law makes a column UNUSABLE rather than permissive: the Δ
semantics of a rule nobody implemented are unknown, not lenient.

Uncertified rows are counted, never dropped
-------------------------------------------
A row whose delta is `NaN` is uncertified: it carries no capture certificate.
Those rows are counted and the fraction travels into the certificate, where it
can trigger a refusal. Silently dropping them would shrink the width by
discarding exactly the data that has no guarantee behind it — a certificate that
gets tighter the less you know is worse than none.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl.execcert import FetchedSeries

#: Laws this module will accept from a caller. Deliberately the verifier's own
#: set, imported rather than restated: two lists of capture laws is how a
#: producer comes to emit one the verifier rejects.
try:                                              # pragma: no cover - trivial
    from alelyon.runtime.atlas.data.attest import KNOWN_CAPTURE_LAWS
except Exception:                                 # pragma: no cover
    KNOWN_CAPTURE_LAWS = frozenset({None, "dither-relative/v0", "exact-cents/v0"})

#: A monetary series stored in whole cents has this step in units of currency.
CENTS = 0.01

Numberish = Union[Sequence[float], np.ndarray, pd.Series]


def _as_series(values: Numberish, name: str) -> pd.Series:
    """Coerce user data to a named float Series with a usable index.

    A default `RangeIndex` is left alone rather than being replaced with dates:
    inventing a calendar the caller did not supply would put a timestamp on
    every figure that nothing stands behind.
    """
    if isinstance(values, pd.Series):
        series = values.astype(float).copy()
    else:
        series = pd.Series(np.asarray(values, dtype=float))
    if series.isna().any():
        raise ValueError(
            f"{name!r} contains NaN. A missing value is not a stored value, so "
            f"it has no quantization step and cannot be certified. Drop it, or "
            f"mark it uncertified with `uncertified_index`")
    series.name = str(name)
    return series


def _as_deltas(delta: Union[float, Numberish], n: int, name: str) -> np.ndarray:
    """Broadcast a scalar step, or validate a per-row one.

    Per-row is the general case and the scalar is the convenience. The guard
    that matters is applied per ELEMENT, not to a summary: a margin check keyed
    on an aggregate passes while individual rows violate it, which is exactly
    how a per-element guarantee gets lost.
    """
    if np.isscalar(delta):
        step = float(delta)
        if not np.isfinite(step) or step < 0:
            raise ValueError(
                f"{name!r}: delta must be finite and non-negative, got {delta!r}")
        return np.full(n, step, dtype=float)
    out = np.asarray(delta, dtype=float)
    if out.shape != (n,):
        raise ValueError(
            f"{name!r}: delta has {out.shape} values for {n} rows; a per-row "
            f"step must have exactly one entry per row")
    finite = np.isfinite(out)
    if np.any(out[finite] < 0):
        raise ValueError(f"{name!r}: a negative quantization step is not a step")
    return out


@dataclass(frozen=True)
class Declared:
    """One series, and what the caller declares about how it was stored."""

    values: Numberish
    #: The quantization step. A scalar broadcasts; a sequence is per-row.
    delta: Union[float, Numberish] = 0.0
    #: The capture law that produced `delta`. See the module docstring.
    law: Optional[str] = None
    #: Positions carrying NO capture certificate. Their delta becomes NaN and
    #: they are COUNTED, which is what lets a certificate refuse rather than
    #: quietly tighten.
    uncertified_index: Sequence[int] = ()

    def fetched(self, name: str) -> FetchedSeries:
        if self.law not in KNOWN_CAPTURE_LAWS:
            raise ValueError(
                f"{name!r}: capture law {self.law!r} is not one this build "
                f"understands ({sorted(x for x in KNOWN_CAPTURE_LAWS if x)}). "
                f"An unknown law is unknown, not lenient: its delta semantics "
                f"are undefined, so the column is unusable rather than accepted")
        series = _as_series(self.values, name)
        deltas = _as_deltas(self.delta, len(series), name)
        for position in self.uncertified_index:
            index = int(position)
            if not 0 <= index < len(deltas):
                raise ValueError(
                    f"{name!r}: uncertified index {index} is outside the "
                    f"{len(deltas)} rows supplied")
            deltas[index] = float("nan")
        uncertified = int(np.count_nonzero(~np.isfinite(deltas)))
        return FetchedSeries(series=series, deltas=deltas,
                             uncertified=uncertified, law=self.law)


class DeclaredFetcher:
    """Serve `certified_run` from data you hold, under a declaration you make.

        from alelyon.runtime.oracle.dsl.fetch import DeclaredFetcher, Declared
        from alelyon.runtime.oracle.dsl.execcert import certified_run

        fetcher = DeclaredFetcher({
            ("price", "ACME"): Declared([100.00, 101.25, 99.50],
                                        delta=0.01, law="exact-cents/v0"),
        })
        cert = certified_run('show mean(price("ACME"))', fetcher=fetcher, seed=7)
        print(cert.base_value, cert.width, cert.level)

    A reference the program makes and this fetcher has no entry for is a
    REFUSAL naming the missing key, not an empty series. An empty series would
    certify a computation over nothing and report a width, which reads as a
    result.
    """

    def __init__(self, declared: Mapping[Tuple[str, str], Declared]) -> None:
        self._declared: Dict[Tuple[str, str], Declared] = {}
        for reference, entry in dict(declared or {}).items():
            if (not isinstance(reference, tuple) or len(reference) != 2
                    or not all(isinstance(part, str) for part in reference)):
                raise ValueError(
                    f"a reference must be (kind, key), got {reference!r}")
            if not isinstance(entry, Declared):
                raise ValueError(
                    f"{reference!r} must map to a Declared, got "
                    f"{type(entry).__name__}")
            self._declared[reference] = entry

    @property
    def references(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self._declared))

    def get(self, kind: str, key: str) -> FetchedSeries:
        entry = self._declared.get((kind, key))
        if entry is None:
            known = ", ".join(f'{k}("{v}")' for k, v in self.references) or "nothing"
            raise ValueError(
                f'the program reads {kind}("{key}") and this fetcher declares '
                f'{known}. Refused rather than served empty: certifying a '
                f'computation over no data still produces a width, which reads '
                f'as a result')
        return entry.fetched(key)


def from_frame(frame: pd.DataFrame, *, delta: Union[float, Mapping[str, float]],
               law: Optional[str] = None, kind: str = "series") -> DeclaredFetcher:
    """A fetcher over every column of a DataFrame.

    `delta` is either one step for all columns or a per-column mapping. A column
    absent from a per-column mapping is an ERROR rather than a zero: a missing
    declaration is missing, and defaulting it to zero would silently assert
    exact storage for the one column nobody thought about.
    """
    declared: Dict[Tuple[str, str], Declared] = {}
    for column in frame.columns:
        name = str(column)
        if isinstance(delta, Mapping):
            if name not in delta:
                raise ValueError(
                    f"no quantization step declared for column {name!r}. "
                    f"Refused rather than defaulted to 0: an absent declaration "
                    f"is absent, and zero would assert exact storage")
            step = delta[name]
        else:
            step = delta
        declared[(kind, name)] = Declared(values=frame[column], delta=step,
                                          law=law)
    return DeclaredFetcher(declared)


def from_csv(path: Union[str, Path], *,
             delta: Union[float, Mapping[str, float]],
             law: Optional[str] = None,
             kind: str = "series",
             index_column: Optional[str] = None) -> DeclaredFetcher:
    """A fetcher over a CSV's numeric columns.

    Non-numeric columns are SKIPPED and named in the error when that leaves
    nothing, rather than being coerced. A column of identifiers silently parsed
    as floats is a column of NaN, and NaN is not a stored value.
    """
    frame = pd.read_csv(path)
    if index_column:
        if index_column not in frame.columns:
            raise ValueError(
                f"{path}: no column {index_column!r}; found "
                f"{list(frame.columns)}")
        frame = frame.set_index(index_column)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(
            f"{path}: no numeric columns. Found {list(frame.columns)}; "
            f"non-numeric columns are skipped rather than coerced, because a "
            f"column of identifiers parsed as floats is a column of NaN")
    return from_frame(numeric, delta=delta, law=law, kind=kind)


__all__ = [
    "CENTS", "Declared", "DeclaredFetcher", "FetchedSeries",
    "from_csv", "from_frame",
]
