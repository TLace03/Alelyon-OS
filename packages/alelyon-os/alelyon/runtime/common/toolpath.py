"""Locate an external program wherever it actually is, not only where PATH says.

`shutil.which` answers one question: *is this on the PATH of the process asking?*
Across this repository that answer was being read as a different one: *is this
installed on the machine?* Those diverge constantly, and they diverge hardest in
exactly the situations that matter:

    a frozen desktop bundle   PyInstaller launches from a Start Menu shortcut with
                              the PATH Explorer hands it, which is the *user* PATH
                              from the registry -- not the developer shell's PATH.
    a GUI child process       a subprocess inherits the launcher's environment; if
                              the launcher was started before an installer amended
                              PATH, the child cannot see the amendment either.
    a fresh install           installers on Windows routinely amend PATH for *new*
                              shells only. Ollama, the GitHub CLI and rustup all
                              do. The program is on disk and unreachable by name.
    a service or scheduler    Task Scheduler and Windows services start with a
                              minimal environment by design.

In every one of those the old code reported `'git' is not on PATH`, which is true
and useless: git is installed, the user can see it, and the program is telling
them to install it again. Worse, the refusals were indistinguishable from genuine
absence, so a real "you have not installed this" was buried in false alarms.

So resolution asks the machine, in this order, and records which one answered:

    OVERRIDE     `$ALELYON_GIT`, `$ALELYON_GH`, ... -- one variable per tool.
                 An explicit answer wins outright, including over a PATH hit,
                 because the only reason to set it is to override a wrong one.
    PATH         `shutil.which`. The fast, ordinary, correct-most-of-the-time case.
    WELL_KNOWN   the documented install locations for this platform, including the
                 per-user ones installers prefer. Versioned directories are
                 globbed and the highest match wins.
    ABSENT       genuinely not on this machine.

Misses are never cached
-----------------------
Hits are cached; misses are not, and the asymmetry is deliberate. `ensure_running`
tells the user "install it, then press Start again" -- caching the miss makes that
instruction a lie for the rest of the process's life. A miss costs a few `stat`
calls, which is the correct price for an answer that must stay true.

`searched` is the point of the refusal
--------------------------------------
Every `Found` carries the list of places that were actually examined, and
`reason()` names them. "not on PATH" tells a user nothing they can act on; "looked
on PATH, in $ALELYON_GH, and in C:\\Program Files\\GitHub CLI\\gh.exe" tells them
precisely which assumption to correct. A refusal a user cannot act on is a bug
report addressed to nobody.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: How the location was determined. Carried on every result so a caller -- or a
#: support transcript -- can tell "found where expected" from "found somewhere
#: PATH never mentioned", which is the difference between a healthy machine and
#: one whose environment is about to cause a different, stranger failure.
OVERRIDE = "override"
PATH = "PATH"
WELL_KNOWN = "well-known"
ABSENT = "absent"

#: Prefix for the per-tool override variables: `git` -> `ALELYON_GIT`.
_ENV_PREFIX = "ALELYON_"


class ToolNotFound(RuntimeError):
    """Raised by `require()`. Carries the `Found` so callers can name the search."""

    def __init__(self, found: "Found") -> None:
        super().__init__(found.reason())
        self.found = found


@dataclass(frozen=True)
class Found:
    """Where a tool is, and what was examined to decide that."""

    tool: str
    path: Optional[str]
    origin: str
    #: Every location examined, in the order examined, as displayable strings.
    #: Present on hits too: knowing a tool was found only after PATH missed is
    #: how an environment problem gets diagnosed before it causes a second bug.
    searched: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.path is not None

    def reason(self) -> str:
        """Why this tool could not be used, naming every place that was looked.

        Empty when it was found -- a caller can treat this as the refusal text
        and get "" exactly when there is nothing to refuse.
        """
        if self.ok:
            return ""
        places = "; ".join(self.searched) if self.searched else "nowhere"
        return (
            f"{self.tool!r} could not be found on this machine. Looked in: "
            f"{places}. If it is installed somewhere else, set "
            f"{env_var(self.tool)} to its full path."
        )


def env_var(tool: str) -> str:
    """The override variable for `tool`: `gh` -> `ALELYON_GH`."""
    stem = Path(tool).stem
    return _ENV_PREFIX + "".join(
        ch if ch.isalnum() else "_" for ch in stem
    ).upper()


# ── where installers actually put things ─────────────────────────────────────
# Entries are templates expanded against the environment. `*` globs, so a
# versioned directory does not need to be enumerated; the highest sorting match
# wins, which for the version-suffixed directories these tools use is the newest.
#
# Ordered per tool from most-specific to least: a per-user install is listed
# before a machine-wide one because when both exist the per-user one is what the
# user most recently chose.
_WINDOWS: Dict[str, Tuple[str, ...]] = {
    "git": (
        r"%ProgramFiles%\Git\cmd\git.exe",
        r"%ProgramFiles(x86)%\Git\cmd\git.exe",
        r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe",
        r"%USERPROFILE%\scoop\shims\git.exe",
        r"%ProgramData%\chocolatey\bin\git.exe",
    ),
    # Git for Windows ships bash; a stock workstation has it installed and not
    # on PATH, which is how the local workflow runner's six shell-step tests
    # were red on the machine that runs them daily. `%SystemRoot%\System32\
    # bash.exe` is deliberately NOT a candidate: that is WSL's launcher, which
    # runs the script inside a Linux distribution where the step's Windows
    # environment (GITHUB_OUTPUT, the workspace path) does not exist as
    # written.
    "bash": (
        r"%ProgramFiles%\Git\bin\bash.exe",
        r"%ProgramFiles(x86)%\Git\bin\bash.exe",
        r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe",
        r"%USERPROFILE%\scoop\shims\bash.exe",
        r"%ProgramData%\chocolatey\bin\bash.exe",
    ),
    "gh": (
        r"%LOCALAPPDATA%\Programs\GitHub CLI\gh.exe",
        r"%ProgramFiles%\GitHub CLI\gh.exe",
        r"%ProgramFiles(x86)%\GitHub CLI\gh.exe",
        r"%USERPROFILE%\scoop\shims\gh.exe",
        r"%ProgramData%\chocolatey\bin\gh.exe",
        # winget installs land under a versioned Packages directory.
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli*\gh.exe",
    ),
    "ollama": (
        r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe",
        r"%ProgramFiles%\Ollama\ollama.exe",
        r"%USERPROFILE%\scoop\shims\ollama.exe",
    ),
    "cargo": (
        r"%CARGO_HOME%\bin\cargo.exe",
        r"%USERPROFILE%\.cargo\bin\cargo.exe",
    ),
    "rustc": (
        r"%CARGO_HOME%\bin\rustc.exe",
        r"%USERPROFILE%\.cargo\bin\rustc.exe",
    ),
    "maturin": (
        r"%USERPROFILE%\.cargo\bin\maturin.exe",
        r"%LOCALAPPDATA%\Programs\Python\Python3*\Scripts\maturin.exe",
    ),
    "node": (
        r"%ProgramFiles%\nodejs\node.exe",
        r"%LOCALAPPDATA%\Programs\nodejs\node.exe",
        r"%USERPROFILE%\scoop\shims\node.exe",
    ),
    "npm": (
        r"%ProgramFiles%\nodejs\npm.cmd",
        r"%LOCALAPPDATA%\Programs\nodejs\npm.cmd",
    ),
    "claude": (
        r"%USERPROFILE%\.local\bin\claude.exe",
        r"%USERPROFILE%\.local\bin\claude.cmd",
        r"%USERPROFILE%\.claude\local\claude.exe",
        r"%APPDATA%\npm\claude.cmd",
        r"%LOCALAPPDATA%\Programs\claude\claude.exe",
    ),
}

#: Paths that carry a tool's NAME without being that tool. A PATH hit landing on
#: one is passed over rather than returned, and recorded in `searched` so the
#: refusal names the namesake it declined.
#:
#: This is the PATH-branch half of the exclusion the `bash` table above already
#: describes. That comment was right about the danger and could not act on it:
#: `find` consults PATH *before* the well-known table and returns on the first
#: hit, so a namesake on PATH is answered with before any table is read. It also
#: named `%SystemRoot%\System32\bash.exe`, which on this workstation does not
#: exist -- the launcher that actually shadows Git's bash is the WindowsApps one
#: below.
#:
#: Windows ships "app execution aliases" in `WindowsApps`: reparse points that
#: launch a Store app. `bash.exe` there is WSL's launcher, and WSL is a different
#: machine wearing the same name. It does not inherit the Windows environment --
#: a workflow step reading `$GITHUB_OUTPUT` gets an empty string and redirects
#: into it -- and it does not accept a Windows path, silently eating the
#: separators, so `C:\Users\...` arrives as `C:Users...: No such file or
#: directory` and the exit code is 127. The same directory holds the `python3`
#: alias stub that once had the pre-push hook print "checks failed; push
#: refused" having run no checks at all.
#:
#: Kept as a filesystem comparison on purpose. `find` promises never to execute a
#: candidate, and a probe would be a larger act than resolution: knowing that a
#: named location is a launcher for something else needs no process.
_NAMESAKES: Dict[str, Tuple[str, ...]] = {
    "bash": (
        r"%SystemRoot%\System32\bash.exe",
        r"%LOCALAPPDATA%\Microsoft\WindowsApps\bash.exe",
    ),
    "sh": (
        r"%SystemRoot%\System32\sh.exe",
        r"%LOCALAPPDATA%\Microsoft\WindowsApps\sh.exe",
    ),
}

_DARWIN: Dict[str, Tuple[str, ...]] = {
    "git": ("/opt/homebrew/bin/git", "/usr/local/bin/git", "/usr/bin/git",
            "/Library/Developer/CommandLineTools/usr/bin/git"),
    "bash": ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"),
    "gh": ("/opt/homebrew/bin/gh", "/usr/local/bin/gh"),
    "ollama": ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama",
               "/Applications/Ollama.app/Contents/Resources/ollama"),
    "cargo": ("$CARGO_HOME/bin/cargo", "$HOME/.cargo/bin/cargo"),
    "rustc": ("$CARGO_HOME/bin/rustc", "$HOME/.cargo/bin/rustc"),
    "maturin": ("$HOME/.cargo/bin/maturin", "/opt/homebrew/bin/maturin"),
    "node": ("/opt/homebrew/bin/node", "/usr/local/bin/node"),
    "npm": ("/opt/homebrew/bin/npm", "/usr/local/bin/npm"),
    "claude": ("$HOME/.local/bin/claude", "$HOME/.claude/local/claude",
               "/opt/homebrew/bin/claude", "/usr/local/bin/claude"),
}

_POSIX: Dict[str, Tuple[str, ...]] = {
    "git": ("/usr/bin/git", "/usr/local/bin/git"),
    "gh": ("/usr/bin/gh", "/usr/local/bin/gh", "/snap/bin/gh"),
    "bash": ("/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"),
    "ollama": ("/usr/local/bin/ollama", "/usr/bin/ollama"),
    "cargo": ("$CARGO_HOME/bin/cargo", "$HOME/.cargo/bin/cargo"),
    "rustc": ("$CARGO_HOME/bin/rustc", "$HOME/.cargo/bin/rustc"),
    "maturin": ("$HOME/.cargo/bin/maturin", "/usr/local/bin/maturin"),
    "node": ("/usr/bin/node", "/usr/local/bin/node"),
    "npm": ("/usr/bin/npm", "/usr/local/bin/npm"),
    "claude": ("$HOME/.local/bin/claude", "$HOME/.claude/local/claude",
               "/usr/local/bin/claude"),
}

#: Tried for any tool with no table entry of its own. The directories a user-level
#: installer picks when it is not being opinionated, so an unlisted tool still
#: gets a real search rather than an immediate refusal.
_GENERIC_WINDOWS = (
    r"%USERPROFILE%\.local\bin\{tool}.exe",
    r"%USERPROFILE%\.local\bin\{tool}.cmd",
    r"%USERPROFILE%\scoop\shims\{tool}.exe",
    r"%ProgramData%\chocolatey\bin\{tool}.exe",
    r"%APPDATA%\npm\{tool}.cmd",
)
_GENERIC_UNIX = (
    "$HOME/.local/bin/{tool}",
    "/usr/local/bin/{tool}",
    "/opt/homebrew/bin/{tool}",
    "/usr/bin/{tool}",
)


def _table() -> Dict[str, Tuple[str, ...]]:
    if sys.platform == "win32":
        return _WINDOWS
    if sys.platform == "darwin":
        return _DARWIN
    return _POSIX


def _generic() -> Tuple[str, ...]:
    return _GENERIC_WINDOWS if sys.platform == "win32" else _GENERIC_UNIX


def _expand(template: str, tool: str) -> Optional[str]:
    """Expand a template against the environment, or None if a variable is unset.

    An unset variable makes the whole candidate meaningless rather than partially
    expanded: `%CARGO_HOME%\\bin\\cargo.exe` with no `CARGO_HOME` must not become a
    search for a literal directory named `%CARGO_HOME%`, which on Windows is a
    legal-but-absent relative path and would be reported as somewhere we looked.
    """
    text = template.replace("{tool}", tool)
    expanded = os.path.expandvars(os.path.expanduser(text))
    if "%" in expanded or "$" in expanded:
        return None
    return expanded


def _executable(path: Path) -> bool:
    """A file we could actually run.

    On POSIX the execute bit is the test. On Windows there is no execute bit and
    `os.access(X_OK)` returns True for any readable file, so being a file is the
    only honest check -- extension filtering happens in the candidate list.
    """
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if sys.platform == "win32":
        return True
    return os.access(str(path), os.X_OK)


def _is_namesake(tool: str, hit: str) -> bool:
    """Whether a PATH hit is a known namesake of `tool` rather than `tool`.

    Compared as paths, never by running anything, so this stays inside `find`'s
    promise. `normcase` because Windows hands back `bash.EXE` where the table
    says `bash.exe`, and a case-sensitive comparison would silently miss.
    """
    if os.name != "nt":
        return False
    target = os.path.normcase(os.path.abspath(hit))
    for template in _NAMESAKES.get(tool, ()):
        expanded = os.path.expandvars(template)
        if "%" in expanded:  # an unset variable; nothing to compare against
            continue
        if os.path.normcase(os.path.abspath(expanded)) == target:
            return True
    return False


def _match(candidate: str) -> Optional[str]:
    """Resolve one candidate, which may contain a `*`. Highest match wins."""
    if "*" not in candidate:
        p = Path(candidate)
        return str(p) if _executable(p) else None
    # `Path.glob` needs the pattern split from a concrete anchor; going through
    # the parent keeps the wildcard confined to the part that has one.
    pattern = Path(candidate)
    anchor = pattern.parent
    while "*" in str(anchor):
        anchor = anchor.parent
    try:
        hits = sorted(
            (p for p in anchor.glob(str(pattern.relative_to(anchor)).replace(os.sep, "/"))
             if _executable(p)),
            key=lambda p: str(p),
        )
    except (OSError, ValueError):
        return None
    return str(hits[-1]) if hits else None


def _candidates(tool: str) -> Tuple[str, ...]:
    """Well-known locations for `tool` on this platform, already expanded."""
    templates = _table().get(tool)
    if templates is None:
        templates = _generic()
    out: List[str] = []
    for template in templates:
        expanded = _expand(template, tool)
        if expanded:
            out.append(expanded)
    return tuple(out)


#: Successful resolutions only -- see the module docstring on why misses are not
#: cached. Keyed by tool name.
_CACHE: Dict[str, Found] = {}


def find(tool: str, *, refresh: bool = False) -> Found:
    """Locate `tool`, recording where it was found and what was examined.

    Never raises and never runs the program -- resolution is a filesystem
    question, and executing a candidate to see whether it works is a far larger
    act than the caller asked for.
    """
    if not refresh:
        cached = _CACHE.get(tool)
        if cached is not None:
            return cached

    searched: List[str] = []

    var = env_var(tool)
    override = os.environ.get(var, "").strip()
    if override:
        searched.append(f"${var}={override}")
        p = Path(os.path.expandvars(os.path.expanduser(override)))
        if _executable(p):
            return _remember(Found(tool, str(p), OVERRIDE, tuple(searched)))
        # An override that points nowhere is a mistake worth surfacing, not a
        # reason to silently fall through to a different program than the one
        # the user named. Recorded in `searched` so the refusal shows it, and
        # the search continues so a working machine is not broken by a typo in
        # a variable someone set months ago.

    searched.append("PATH")
    hit = shutil.which(tool)
    if hit and _is_namesake(tool, hit):
        # Recorded rather than dropped: a machine where Git's bash is missing
        # entirely should refuse with "I found WSL's launcher and would not use
        # it", which is actionable, instead of a bare "not found" that argues
        # with the user's own `where bash`.
        searched.append(f"{hit} (a namesake, not {tool})")
        hit = None
    if hit:
        return _remember(Found(tool, hit, PATH, tuple(searched)))

    for candidate in _candidates(tool):
        searched.append(candidate)
        resolved = _match(candidate)
        if resolved:
            return _remember(Found(tool, resolved, WELL_KNOWN, tuple(searched)))

    # Deliberately NOT cached.
    return Found(tool, None, ABSENT, tuple(searched))


def _remember(found: Found) -> Found:
    _CACHE[found.tool] = found
    return found


def which(tool: str) -> Optional[str]:
    """Drop-in for `shutil.which` that also searches the well-known locations."""
    return find(tool).path


def available(tool: str) -> bool:
    return find(tool).ok


def require(tool: str) -> str:
    """The absolute path, or raise `ToolNotFound` naming everywhere searched."""
    found = find(tool)
    if not found.ok:
        raise ToolNotFound(found)
    return found.path  # type: ignore[return-value]


def argv(tool: str, *args: str) -> List[str]:
    """`[<resolved path>, *args]`, falling back to the bare name.

    The fallback is intentional: a caller that has already decided to proceed --
    typically because it handles the failure itself -- should get the operating
    system's own error rather than a different one invented here. Callers that
    want to refuse cleanly ask `find()` first and use `reason()`.
    """
    return [find(tool).path or tool, *args]


#: Windows: do not give the child a console of its own.
_CREATE_NO_WINDOW = 0x08000000


def no_window() -> dict:
    """Keyword arguments that stop a child process flashing a console window.

    Splat into every `subprocess` call that runs a CONSOLE tool — `git`, `npm`,
    `python` — from code that can run inside the desktop application::

        subprocess.run(toolpath.argv("git", "status"), **toolpath.no_window())

    Why this is not cosmetic. The packaged application is built with
    PyInstaller's windowed bootloader (`runw.exe`), so the process owns NO
    console. On Windows a console child of a console-less parent does not
    inherit one — it ALLOCATES ITS OWN, which appears as a terminal window.
    The Fleet views poll on timers and shell out to `git` once per worktree per
    tick, so the installed application showed terminal windows opening and
    closing continuously, forever, with nothing visibly running in Task Manager
    because each child lived only milliseconds.

    It is empty on POSIX, where the flag does not exist and no window is
    created either way, so call sites need no platform branch of their own.
    `engine_lifecycle` and `supervisor` keep their own explicit flags: they also
    want DETACHED_PROCESS and a new process group, which is a different
    intention from "do not flash a window".
    """
    if sys.platform == "win32":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


def clear_cache() -> None:
    """Forget resolved locations. For tests, and for after an install."""
    _CACHE.clear()


def report(tools: Optional[Sequence[str]] = None) -> Tuple[Found, ...]:
    """Resolve several tools at once, for a diagnostics view.

    Bypasses the cache so a report reflects the machine now rather than whatever
    was true the first time each tool happened to be asked for.
    """
    names: Iterable[str] = tools if tools is not None else sorted(_table())
    return tuple(find(name, refresh=True) for name in names)


__all__ = [
    "ABSENT", "OVERRIDE", "PATH", "WELL_KNOWN",
    "Found", "ToolNotFound",
    "argv", "available", "clear_cache", "env_var", "find", "report",
    "require", "which",
]
