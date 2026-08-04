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
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: Directory names that mean "this module was installed, not checked out".
_INSTALL_MARKERS = ("site-packages", "dist-packages")

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


def _resolve() -> tuple[Path, Path, bool]:
    """Return (repo root or state root, state home, whether this is installed)."""
    env = os.environ.get("ALELYON_HOME")
    if env:
        root = Path(env).expanduser().resolve()
        return root, root / "globals", False

    here = Path(__file__).resolve()
    if not _is_installed(here):
        for parent in here.parents:
            if (parent / "pyproject.toml").is_file():
                return parent, parent / "globals", False

    # Installed, or a checkout with no pyproject.toml above this module. Either
    # way there is no repository to anchor state to, and inventing one from the
    # module's own nesting depth writes into whatever directory happens to be
    # there. A per-user directory is the honest answer.
    state = _user_state_dir()
    return state, state, True


REPO_ROOT, GLOBALS_DIR, INSTALLED = _resolve()

#: True when no source checkout backs these paths — `REPO_ROOT` is then a state
#: directory rather than a repository, and nothing may assume a repository
#: layout (`<root>/alelyon/...`, `<root>/docs/...`) exists beneath it.
__all__ = ["REPO_ROOT", "GLOBALS_DIR", "INSTALLED"]
