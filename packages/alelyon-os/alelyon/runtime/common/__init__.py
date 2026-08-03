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


def __getattr__(name):
    target = _LAZY_TARGETS.get(name)
    if target is None:
        raise AttributeError(f"module 'alelyon.runtime.common' has no attribute {name!r}")
    import importlib
    obj = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = obj      # cache on the package for instant subsequent access
    return obj


def __dir__():
    return list(globals().keys()) + list(_LAZY_TARGETS.keys())
