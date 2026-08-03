"""Seam to the native deterministic kernels (alelyon-vector, Rust/PyO3).

The DQC-OS split of labor: the certified pipeline bounds the DATA error;
`alelyon_vector` removes COMPUTE-side nondeterminism — fixed-order Neumaier
reductions that are bit-identical across runs, thread counts, and machines
(multi-threaded, SIMD-reassociating BLAS is not). The dither stream stays in
Python (certkit's seeded generator), so the Rust quantizer core is BIT-PARITY
tested against certkit end to end.

Soft dependency: everything here falls back to NumPy when the extension isn't
installed (`pip install alelyon/languages/vector_native/target/wheels/<whl>`,
or `maturin build --release -i <python>` to produce one). `available()` tells
callers which substrate they're on; `det_*` results on the fallback path are
NOT bit-reproducible across BLAS configurations — only the native path is.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import alelyon_vector as _native
except ImportError:                       # pragma: no cover - environment-dependent
    _native = None


def available() -> bool:
    return _native is not None


def version() -> str:
    return getattr(_native, "__version__", "absent") if _native else "absent"


def det_sum(x: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if _native is not None:
        return float(_native.det_sum(x))
    return float(np.sum(x))


def det_dot(x: np.ndarray, y: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if _native is not None:
        return float(_native.det_dot(x, y))
    return float(np.dot(x, y))


def det_mean_var(x: np.ndarray) -> Tuple[float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if _native is not None:
        m, v = _native.det_mean_var(x)
        return float(m), float(v)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.var(ddof=1))


def det_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if _native is not None:
        return float(_native.det_corr(x, y))
    if x.size != y.size or x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def dither_quantize(x: np.ndarray, u: np.ndarray, delta: float) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    if _native is not None:
        return np.asarray(_native.dither_quantize(x, u, delta))
    q = np.round((np.where(np.isfinite(x), x, 0.0) + u) / float(delta))
    # match the Rust kernel's SATURATING float->i64 cast (NumPy's astype wraps
    # modularly, which would diverge from native on an over-range code — the
    # certify layer keeps codes in range, so this only matters for direct callers)
    ii = np.iinfo(np.int64)
    return np.clip(q, float(ii.min), float(ii.max)).astype(np.int64)


def dither_reconstruct(q: np.ndarray, u: np.ndarray, delta: float) -> np.ndarray:
    q = np.ascontiguousarray(q, dtype=np.int64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    if _native is not None:
        return np.asarray(_native.dither_reconstruct(q, u, delta))
    return q.astype(np.float64) * float(delta) - u
