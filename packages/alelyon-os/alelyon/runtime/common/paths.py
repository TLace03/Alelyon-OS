"""Depth-independent repo-root + data-dir resolution for runtime.common.

`REPO_ROOT` is discovered WITHOUT relying on this file's nesting depth — it walks up to
the directory containing `pyproject.toml`, or honors `$ALELYON_HOME` — so modules under
`alelyon.*` can move freely without breaking file I/O.

`GLOBALS_DIR` is the on-disk state home. Inside a source checkout it is `<repo>/globals`,
unchanged: the migration moved the *code* out of `globals/` but deliberately left the
*state* (databases, caches, sentinels, JSON) in place, and Phase 2b relocates that state
under `var/`. Keeping the anchor here means every path constant resolves to exactly
where it did before the move.

Installed, there is no checkout to anchor to
--------------------------------------------
`alelyon-os` publishes modules that write state — `worktree_cache.default_database()`
puts the fleet's SQLite file under `GLOBALS_DIR`, and the assistant's conversation
history goes there too. When the package is installed from a wheel there is no
`pyproject.toml` above it, so the old fallback (`here.parents[3]`) resolved to whatever
directory happened to sit three levels above the module. Inside a virtualenv that is
`site-packages`, which means a `pip install alelyon-os` followed by `alelyon-fleet
status` created `…/site-packages/globals/worktree_cache.db`: state written into the
package directory, lost on upgrade, invisible to the user, and shared between every
project on the machine.

So the resolution now distinguishes the two cases by name rather than by accident:

    ALELYON_HOME set      that directory, whatever it is. The explicit answer wins.
    a real checkout       the directory holding `pyproject.toml`. Unchanged.
    installed             a per-user state directory for the platform, created on
                          demand. `INSTALLED` records which of these happened.

`site-packages` is checked explicitly rather than being caught by "no pyproject.toml
found": a wheel installed into a directory that happens to sit under some unrelated
project's checkout would otherwise adopt that project's `globals/` and write another
program's state into it.

Resolution is LAZY, and why that is not a style choice
-----------------------------------------------------
`REPO_ROOT, GLOBALS_DIR, INSTALLED` used to be assigned at module import and never
recomputed. `alelyon.platform.deployment.runtime_env.bootstrap()` publishes
`$ALELYON_HOME` and is documented as "the FIRST thing an entry point does" — but a
module that imported this one before `bootstrap()` ran froze the pre-bootstrap answer
for the life of the process, and the two answers disagree about the `globals/`
component. Both were populated on this workstation: measured 2026-08-11, one
`fleet_repository_paths.sqlite3` at the state-root top level and a second under
`<state root>/globals/`, written the same day nine hours apart. Two live stores with
divergent contents is the failure that produces.

So the three names are served by a module `__getattr__` over a resolver cached on the
environment it depends on (`ALELYON_HOME`, `ALELYON_FORCE_PACKAGED`). Reading them
still looks exactly the same to every existing importer; the difference is that a
`bootstrap()` which runs after this module was imported is now seen rather than
ignored. Names bound by `from ... import GLOBALS_DIR` are still snapshots at *that*
module's import — that is a Python binding rule, not something this file can change —
which is why `default_database()`-style callers should read through `globals_dir()`.

The two branches also had to be made to AGREE. `ALELYON_HOME` resolves to
`<root>/globals`, and `bootstrap()` creates exactly that directory; the installed
branch returned `<state root>` with no `globals/` component at all. It now returns
`<state root>/globals`, which is the same directory `bootstrap()` prepares.

`ALELYON_FORCE_PACKAGED` is honoured here as well. `runtime_env` documents it as the
way to reproduce a packaged layout from an unfrozen interpreter, and this module not
consulting it meant the packaged-layout smoke test resolved SOURCE paths and could not
detect this bug class at all.

Fleet-wide state is anchored on the git common dir
--------------------------------------------------
`GLOBALS_DIR` is "the directory holding `pyproject.toml`", which a linked worktree
satisfies too — and there are 287 of them here, each with its own. That is right for
state a checkout owns and wrong for state the *fleet* shares. `fleet_state_dir()` is
the shared answer, keyed on `git rev-parse --git-common-dir`, which every worktree of
one repository reports identically. This is not a new decision: `pr_relay` and
`worktree_cache.default_database()` already anchor there, for the same reason and with
the same words.
"""
from __future__ import annotations

import os
import sys
import sys as _sys
import types as _types
from pathlib import Path

#: Directory names that mean "this module was installed, not checked out".
_INSTALL_MARKERS = ("site-packages", "dist-packages")

#: The explicit state-root override. `runtime_env.STATE_ROOT_ENV` is the same name;
#: it is spelled literally here because this module is the runtime floor and cannot
#: import the packaging layer that publishes it.
_HOME_ENV = "ALELYON_HOME"

#: Opt into packaged path rules from an unfrozen interpreter. Same variable as
#: `runtime_env.FORCE_PACKAGED_ENV`, and honoured here so that a smoke test which
#: sets it actually exercises packaged path resolution.
_FORCE_PACKAGED_ENV = "ALELYON_FORCE_PACKAGED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: How long a `git rev-parse` may take before the fleet anchor degrades to the
#: per-checkout answer. Matches the sibling modules that already shell out to git
#: (`worktree.py`, `worktree_cache.py`, `fleet_outcomes.py` use 30-60s).
_GIT_TIMEOUT = 30.0

#: Where per-user state goes when there is no checkout. One directory per
#: platform convention, so the answer is where that platform's users look.
_APP_DIR = "Alelyon"


def _is_installed(path: Path) -> bool:
    return any(part in _INSTALL_MARKERS for part in path.parts)


def _user_state_dir() -> Path:
    """The platform's per-user application-data directory for this package.

    Deliberately NOT a dot-directory in `$HOME` on Windows or macOS: each of
    those platforms has a documented location, and putting state somewhere else
    means the user cannot find it, back it up, or clear it by the means their
    system already gives them.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / _APP_DIR
        return Path.home() / "AppData" / "Local" / _APP_DIR
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIR
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _APP_DIR.lower()
    return Path.home() / ".local" / "share" / _APP_DIR.lower()


def user_state_home() -> Path:
    """The per-user state home — `globals/` component INCLUDED.

    `_user_state_dir()` is the platform directory; this is the state home inside
    it, and the difference between the two is exactly one component and one
    divergent database. A caller that needs per-user state in a SOURCE checkout
    cannot use `globals_dir()` (which anchors on the checkout) and so reached for
    `_user_state_dir()` instead — landing one component above every other branch
    of `_resolve()`. Measured on this workstation 2026-08-11, that produced two
    live `fleet_repository_paths.sqlite3` stores with divergent contents:
    106,496 B at the platform directory and 40,960 B under its `globals/`.

    So the component is defined ONCE, here, and `_resolve()`'s installed branch
    is written in terms of it. The two cannot drift apart again by being edited
    separately, which is how they drifted apart in the first place.
    """
    return _user_state_dir() / "globals"


def _forced_packaged() -> bool:
    """True when `$ALELYON_FORCE_PACKAGED` asks for packaged path rules.

    Kept byte-compatible with `runtime_env.is_packaged()` so the two cannot drift
    into disagreeing about what the same variable means.
    """
    return (os.environ.get(_FORCE_PACKAGED_ENV) or "").strip().lower() in _TRUTHY


def _resolve() -> tuple[Path, Path, bool]:
    """Return (repo root or state root, state home, whether this is installed).

    Every branch ends in a `globals/` component. The installed branch used to
    return the state root itself, which meant a process that ran `bootstrap()`
    and a process that did not wrote to two different directories under the same
    name — see the module docstring for the measurement.
    """
    env = os.environ.get(_HOME_ENV)
    if env and env.strip():
        root = Path(env).expanduser().resolve()
        return root, root / "globals", False

    if _forced_packaged():
        # An unfrozen interpreter that asked to be treated as packaged. Honouring
        # it here is what lets the packaged-layout smoke test resolve packaged
        # paths instead of quietly measuring the source checkout.
        return _user_state_dir(), user_state_home(), True

    here = Path(__file__).resolve()
    if not _is_installed(here):
        for parent in here.parents:
            if (parent / "pyproject.toml").is_file():
                return parent, parent / "globals", False

    # Installed, or a checkout with no pyproject.toml above this module. Either
    # way there is no repository to anchor state to, and inventing one from the
    # module's own nesting depth writes into whatever directory happens to be
    # there. A per-user directory is the honest answer.
    return _user_state_dir(), user_state_home(), True


# ── Lazy resolution ──────────────────────────────────────────────────────────
# Cached on the environment the answer depends on, so `bootstrap()` publishing
# `$ALELYON_HOME` after this module was imported is seen rather than ignored.

_RESOLVED: tuple[Path, Path, bool] | None = None
_RESOLVED_KEY: tuple[str | None, str | None] | None = None


def _environment_key() -> tuple[str | None, str | None]:
    return (os.environ.get(_HOME_ENV), os.environ.get(_FORCE_PACKAGED_ENV))


def resolve(*, refresh: bool = False) -> tuple[Path, Path, bool]:
    """`(REPO_ROOT, GLOBALS_DIR, INSTALLED)`, recomputed when the env changed."""
    global _RESOLVED, _RESOLVED_KEY
    key = _environment_key()
    if refresh or _RESOLVED is None or _RESOLVED_KEY != key:
        _RESOLVED = _resolve()
        _RESOLVED_KEY = key
    return _RESOLVED


def repo_root() -> Path:
    """The checkout root, or the state root when there is no checkout."""
    return resolve()[0]


def globals_dir() -> Path:
    """The on-disk state home. Prefer this over the `GLOBALS_DIR` snapshot.

    A module-level `from ... import GLOBALS_DIR` binds one value at that module's
    import; calling this binds nothing and always reflects the current answer.
    """
    return resolve()[1]


def installed() -> bool:
    """True when no source checkout backs these paths."""
    return resolve()[2]


_LAZY_NAMES = {"REPO_ROOT": 0, "GLOBALS_DIR": 1, "INSTALLED": 2}

#: Explicit overrides a CALLER assigned, by name. Separate storage from the
#: module globals, so restoring a value cannot be mistaken for assigning one.
_OVERRIDES: dict[str, object] = {}

#: The exact object last handed out for each name, kept BY IDENTITY. See
#: `_PathsModule` for why identity rather than equality.
_HANDED_OUT: dict[str, object] = {}


class _PathsModule(_types.ModuleType):
    """Serves the three historical constants through a property, not PEP 562.

    A module `__getattr__` cannot hold this on its own, and the way it fails
    looks exactly like a passing test. PEP 562 fires only when normal attribute
    lookup FAILS. `monkeypatch.setattr(paths, "GLOBALS_DIR", x)` reads the old
    value first — through `__getattr__`, since there is no real global — and on
    teardown restores it by SETTING it. That creates a real module global, which
    from then on satisfies normal lookup, so `__getattr__` is never consulted
    again and the environment-keyed cache below it is dead for the life of the
    process. `tests/runtime/test_worktree_bus.py` patches this name, and after it
    ran, `test_the_module_constants_follow_a_later_bootstrap` in
    `test_fleet_store_resilience.py` read the frozen answer and failed —
    MEASURED, and it passed in isolation, which is what made it look like flake.

    A restore is told from an assignment BY IDENTITY: if the value being set is
    the very object this module last handed out for that name, it is an undo and
    clears the override rather than pinning it. Identity rather than equality
    because these are a `Path`, a `Path` and a `bool`, and `bool` cannot be
    subclassed to carry a marker the way `db._DB_PATH` does. Assigning the
    identical object the module just produced is a no-op override in any case,
    so treating it as one is not a lost capability.
    """

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails, which is every read of these
        # three, because nothing ever writes them into the module globals.
        index = _LAZY_NAMES.get(name)
        if index is None:
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}")
        if name in _OVERRIDES:
            return _OVERRIDES[name]
        value = resolve()[index]
        _HANDED_OUT[name] = value
        return value

    def __setattr__(self, name: str, value) -> None:
        if name in _LAZY_NAMES:
            if _HANDED_OUT.get(name) is value:
                _OVERRIDES.pop(name, None)      # an undo, not a redirect
            else:
                _OVERRIDES[name] = value
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in _LAZY_NAMES:
            _OVERRIDES.pop(name, None)
            return
        super().__delattr__(name)


_sys.modules[__name__].__class__ = _PathsModule


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_NAMES))


def fleet_state_dir() -> Path:
    """State shared by every worktree of ONE repository.

    `GLOBALS_DIR` answers "state this checkout owns", and a linked worktree is a
    real checkout with a `pyproject.toml` of its own — so anchoring fleet-wide
    state there gives each of the 287 worktrees on this workstation a private
    copy of a store whose entire purpose is to be shared. Git already knows the
    right scope: every worktree of a repository reports the same
    `--git-common-dir`.

    Degrades to `globals_dir()` rather than raising, and the degradation is
    ordinary rather than exceptional: an installed wheel has no git and no
    checkout, and that case was never the broken one. A git failure degrades the
    same way — a private store is worse than a shared one, and no store at all is
    worse than both.
    """
    root, state, is_installed = resolve()
    if is_installed or (os.environ.get(_HOME_ENV) or "").strip():
        # An explicit state root wins outright, as it does everywhere else. It
        # may sit inside some unrelated checkout, and adopting that checkout's
        # git anchor would write this program's state into another project.
        return state
    shared = _git_common_root(root)
    if shared is None:
        return state
    return shared / "globals"


_FLEET_ANCHORS: dict[str, Path | None] = {}


def _git_common_root(start: Path) -> Path | None:
    """The primary checkout of the repository containing `start`, or None.

    Cached per starting directory: this shells out, and `fleet_state_dir()` is
    called on read paths that a view may poll.
    """
    key = os.path.normcase(str(start))
    if key in _FLEET_ANCHORS:
        return _FLEET_ANCHORS[key]
    answer: Path | None = None
    try:
        import subprocess

        from alelyon.runtime.common import toolpath

        proc = subprocess.run(
            toolpath.argv("git", "-C", str(start), "rev-parse",
                          "--path-format=absolute", "--git-common-dir"),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GIT_TIMEOUT, **toolpath.no_window())
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            common = Path(out).resolve()
            # `<primary>/.git` for a checkout and for every worktree of it.
            parent = common.parent
            if parent.exists():
                answer = parent
    except Exception:                                              # noqa: BLE001
        # git absent, not a repository, timed out, or the toolpath layer is not
        # importable in this process. Every one of those means "no shared
        # anchor", which the caller degrades on.
        answer = None
    _FLEET_ANCHORS[key] = answer
    return answer


def _package_root() -> Path:
    """The `alelyon` package directory, found by NAME rather than by depth.

    `REPO_ROOT` answers "where does state live", and deliberately becomes a
    per-user directory when installed. A different question keeps being asked
    across the tree -- "where is the code I shipped with" -- and it kept being
    answered by counting parents:

        Path(__file__).resolve().parents[3]        # frontend/desktop/...
        Path(__file__).resolve().parents[5]        # frontend/desktop/lattice/...
        Path(__file__).resolve().parent.parent     # and this one was WRONG

    Every one of those encodes the module's own nesting depth at the moment it
    was written, which is precisely what breaks when a file moves. The
    `alelyon.*` refactor moved files and silently invalidated the counts; the
    `parent.parent` above resolved to `alelyon/frontend/globals/`, a directory
    that has never existed, and the cache written through it failed into a bare
    `except Exception` for months without one visible symptom.

    A package's relationship to the modules inside it does not depend on where
    the tree as a whole is mounted, so walking up to the directory NAMED
    `alelyon` is stable under a checkout, a wheel, and a frozen bundle alike.

    Nearest match wins, which matters for a checkout that is itself in a
    directory called `alelyon`: `.../alelyon/alelyon/runtime/...` must resolve to
    the inner one, and `Path.parents` is ordered nearest-first.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "alelyon":
            return parent
    # This module is inside the package, so the loop finding nothing means the
    # package was renamed. Its own directory is the closest honest answer.
    return here.parent


#: The `alelyon` package directory itself.
PACKAGE_ROOT = _package_root()

#: The directory CONTAINING the package -- the repository root in a checkout,
#: `site-packages` in an installed wheel, the bundle directory when frozen.
#: Use for locating code and shipped assets. For STATE, use `GLOBALS_DIR`.
INSTALL_ROOT = PACKAGE_ROOT.parent

#: `INSTALLED` is True when no source checkout backs these paths — `REPO_ROOT`
#: is then a state directory rather than a repository, and nothing may assume a
#: repository layout (`<root>/alelyon/...`, `<root>/docs/...`) exists beneath it.
#:
#: `REPO_ROOT`, `GLOBALS_DIR` and `INSTALLED` are served lazily by `__getattr__`
#: and are not module globals. They read identically; what changed is that they
#: are recomputed when `$ALELYON_HOME` or `$ALELYON_FORCE_PACKAGED` changes.
__all__ = ["REPO_ROOT", "GLOBALS_DIR", "INSTALLED", "PACKAGE_ROOT",
           "INSTALL_ROOT", "fleet_state_dir", "globals_dir", "installed",
           "repo_root", "resolve"]
