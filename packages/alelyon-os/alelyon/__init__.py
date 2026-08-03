"""Alelyon — the single distribution namespace for all platform code.

Nesting every pillar under `alelyon.` resolves the top-level name collisions
(notably `platform` shadowing the stdlib, plus `vector`/`data` recurring across
the tree). Import paths are `alelyon.runtime.vector`, `alelyon.platform.telemetry`,
etc. See docs/ARCHITECTURE.md.
"""
