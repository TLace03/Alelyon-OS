"""Concrete tool packs, grouped by the desk they read.

These are the **Financial Markets** domain's packs (`calc` is also the general
domain's, being the one capability here that is about computing a figure rather
than about markets). A pack is a module exposing `install(registry)`, which
registers its tools into the registry it was handed — never into a shared one.
`domain.Domain.toolset()` is what calls it, once per domain, and a pack that
fails to import loses its own tools and is named by `Domain.failed_packs()`
rather than taking the catalogue down with it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from alelyon.runtime.oracle.assistant.tools import Registry

#: Every markets pack, in the order the router's catalogue lists them. Kept as
#: a name so `domain.MARKETS` and the test that asserts the set stay in step.
_MODULES = ("book", "market", "derivatives", "macro", "company",
            "engine_state", "calc")


def install_all(registry: Registry,
                modules: Optional[Sequence[str]] = None) -> List[str]:
    """Install packs into `registry`. Returns the names of any that failed.

    Kept as a convenience for a caller that wants the markets packs without a
    `Domain`; the shipped path is `Domain.toolset()`, which does the same thing
    against its own registry and caches the result.
    """
    failed: List[str] = []
    for name in (modules if modules is not None else _MODULES):
        try:
            mod = __import__(f"alelyon.runtime.oracle.assistant.desks.{name}",
                             fromlist=["install"])
            mod.install(registry)
        except Exception:  # noqa: BLE001
            failed.append(name)
    return failed
