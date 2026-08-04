"""Common — shared low-level utilities (config, constants, paths, logging, atomic IO).

The floor of the runtime dependency graph: depends on nothing else in `alelyon.runtime`.
Formerly the top-level `globals` package (now a deprecation shim — see `globals/`).
"""
# Lazy re-exports: `from alelyon.runtime.common import RuntimeConfig` (and the legacy
# `from alelyon.runtime.common import RuntimeConfig` via the shim) keep working without eagerly
# importing the numpy/threading-heavy modules — so `python main.py --status` stays
# runnable on a barebones interpreter.
_LAZY_TARGETS = {
    "RuntimeConfig":       ("alelyon.runtime.common.config_globals",       "RuntimeConfig"),
    "load_runtime_config": ("alelyon.runtime.common.config_globals",       "load_runtime_config"),
    "TRADABLE_UNIVERSE":   ("alelyon.runtime.common.constants_globals",    "TRADABLE_UNIVERSE"),
    "_NumpyEncoder":       ("alelyon.runtime.common.common_utils_globals", "_NumpyEncoder"),
    "_ny_now":             ("alelyon.runtime.common.common_utils_globals", "_ny_now"),
    "_is_ny_session":      ("alelyon.runtime.common.common_utils_globals", "_is_ny_session"),
    "atomic_write_json":   ("alelyon.runtime.common.common_utils_globals", "atomic_write_json"),
}


#: This package ships in the PUBLIC `alelyon-os` wheel (the `fleet` subsystem),
#: which carries `worktree*.py` and this file — and none of the modules
#: `_LAZY_TARGETS` points at. So in a `pip install alelyon-os` every one of
#: those seven names resolved to `ModuleNotFoundError`, `dir()` advertised all
#: seven as though they worked, and the error text published the names of three
#: private modules (BT-OS-001, `docs/BLUE_TEAM_2026-07-31.md` §21.3).
#:
#: The table stays the single source of truth and the file stays byte-identical
#: between this tree and the wheel — a generated variant would break exactly the
#: property the export audit relies on. What changes is that the ADVERTISED
#: surface is now what this distribution can actually serve.


def _lazy_available(target: tuple) -> bool:
    """Whether a lazy target's module is present in THIS distribution.

    Failing toward "available" is deliberate. This decides only what `dir()`
    advertises, and the cost of being wrong in the two directions is not
    symmetric: hiding a name that works would make a frozen or zipped build
    look broken, while showing one that does not is repaired by `__getattr__`
    below raising a clean `AttributeError`.
    """
    try:
        import importlib.util
        return importlib.util.find_spec(target[0]) is not None
    except Exception:  # noqa: BLE001 - a finder that raises is not an answer
        return True


def __getattr__(name):
    target = _LAZY_TARGETS.get(name)
    if target is None:
        raise AttributeError(f"module 'alelyon.runtime.common' has no attribute {name!r}")
    import importlib
    try:
        module = importlib.import_module(target[0])
    except ModuleNotFoundError as exc:
        # The module this distribution does not carry. Two things matter here:
        # `AttributeError` is the contract a missing module attribute owes its
        # caller (and it is what `hasattr` and `getattr(..., default)` expect),
        # and the message must not name the private module — publishing the
        # path of something deliberately withheld is the other half of the
        # defect. A ModuleNotFoundError from DEEPER in the import is re-raised
        # untouched: that is a broken dependency, not an absent surface.
        if (exc.name or "") != target[0]:
            raise
        raise AttributeError(
            f"module 'alelyon.runtime.common' has no attribute {name!r} in "
            f"this distribution") from None
    obj = getattr(module, target[1])
    globals()[name] = obj      # cache on the package for instant subsequent access
    return obj


def __dir__():
    # DEDUPLICATED, and that is not tidiness. `__getattr__` above caches a
    # resolved name into `globals()`, so after any code path has touched one of
    # these seven it appears in BOTH halves of this list — and `dir()` sorts its
    # result without deduplicating it. The advertised surface is checked by
    # comparing `dir()` against the expected set, so a duplicate breaks that
    # check for every caller, while the module is working perfectly.
    #
    # It stayed latent because it only fires once something has resolved a lazy
    # name in the same process: the packaging test passed alone and failed after
    # `tests/atlas` or `tests/integration` ran first.
    return sorted({
        *globals(),
        *(name for name, target in _LAZY_TARGETS.items()
          if _lazy_available(target)),
    })
