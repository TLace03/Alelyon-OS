"""FAM data layer — the keystone: one front door + a write-through history store.

    from alelyon.runtime.atlas.data import data_service, DataService, HistoryStore, default_store

Going forward, analytics engines should depend on DataService (caching +
recording + a prefer_store read path) rather than calling yfinance/FRED directly.
History accrues from the day this ships and can't be bought back later.
"""
__all__ = ["data_service", "DataService", "HistoryStore", "default_store"]


# LAZY re-exports (PEP 562): keep the convenience API
#   from alelyon.runtime.atlas.data import data_service, HistoryStore
# working, but DON'T eagerly pull the store/service when a caller only wants a
# PURE sibling module (e.g. `attest`, the open verifier's crypto/Merkle core).
# That keeps the alelyon.verify import closure free of the engine/store — the
# precondition for shipping the verifier as a standalone open-source library.
def __getattr__(name):
    if name in ("HistoryStore", "default_store"):
        from alelyon.runtime.atlas.data import history
        return getattr(history, name)
    if name in ("DataService", "data_service"):
        from alelyon.runtime.atlas.data import service
        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
