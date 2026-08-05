"""The local model, managed from inside the app.

The AI Analyst assumed Ollama was already running with the right model pulled.
When it was not — which is every fresh machine, and any machine after a reboot —
the only symptom was an answer that never arrived, and the fix required leaving
the application to run shell commands. For a feature meant to be *available on
demand*, "start a server first" is not on demand.

This module makes the model's state a first-class thing the app can see and act
on. It is deliberately thin: probe, start, list, pull, select. It is not a model
manager and should not become one.

**Four states, four different sentences.** Collapsing them is how "the analyst
is broken" gets reported for four unrelated causes:

    OFFLINE     the Ollama server is not reachable       → start it
    NO_MODEL    the server is up; the model is not pulled → download it
    READY       server up, model present                  → ask away
    ERROR       the server answered something unusable    → show what it said

**Nothing here starts a download on its own.** A model is gigabytes over
someone's connection, possibly metered. `ensure_running` will start a server
that is already installed, because that is instant and local; pulling weights is
always an explicit act with a visible size.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from alelyon.runtime.common import toolpath
from alelyon.runtime.common.paths import GLOBALS_DIR

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_BASE = "http://localhost:11434"

STATE_OFFLINE = "offline"
STATE_NO_MODEL = "no_model"
STATE_READY = "ready"
STATE_ERROR = "error"

_PROBE_TIMEOUT = 2.0        # a UI probe must never hang the panel
_START_GRACE = 12.0         # how long a freshly-spawned server gets to listen

#: How long to spend deciding whether anything is listening at all, per address.
#:
#: This is NOT the same budget as `_PROBE_TIMEOUT`, and the distinction is the
#: whole point. `_PROBE_TIMEOUT` bounds one HTTP request; a probe makes two of
#: them, and `socket.create_connection` applies its timeout PER ADDRESS returned
#: by `getaddrinfo`. `localhost` resolves to both `::1` and `127.0.0.1` on
#: Windows, so the honest arithmetic on a machine with no Ollama was
#: 2.0s x 2 addresses x 2 requests = 8 seconds — measured, on the UI thread,
#: inside `MainWindow.__init__`. That was the hang on a fresh install.
#:
#: A server on loopback either answers a TCP connect in single-digit
#: milliseconds or is not there. 0.35s is generous for that question and cannot
#: accumulate, because `_server_listening()` stops at the first success.
_CONNECT_TIMEOUT = 0.35

#: How many addresses one loopback host costs. `localhost` resolves to both
#: `::1` and `127.0.0.1`, and both `socket.create_connection` and `urlopen`
#: apply their timeout per address, so every budget below is paid this many
#: times. Named rather than written as a bare `2` because it is the multiplier
#: that made the arithmetic above wrong the first time.
_ADDRESSES_PER_HOST = 2

#: Worst-case wall time of one `probe()` call, in seconds: the listen gate plus
#: the two HTTP requests, each paid per address. Currently 8.7s.
#:
#: Public because a caller that has to *join* a probe running on a thread needs
#: this budget and must not re-derive it. A `QThread` still running when it is
#: destroyed aborts the process (0xC0000409), so a join bound that sits below
#: the real worst case is not a slow join — it is the crash. Widening either
#: timeout above widens this with it.
PROBE_WORST_CASE = (_CONNECT_TIMEOUT + 2 * _PROBE_TIMEOUT) * _ADDRESSES_PER_HOST

_LOCK = threading.Lock()


def base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE).rstrip("/")


# The chosen model is runtime state, not a GUI preference: the API server and
# any headless caller need it too, so it cannot live in QSettings.
_PREF_PATH = GLOBALS_DIR / "analyst_model.json"


def selected_model() -> str:
    """The model the analyst will use. Env wins, so a machine can override
    without touching stored state."""
    env = os.environ.get("OLLAMA_MODEL")
    if env:
        return env
    try:
        raw = json.loads(_PREF_PATH.read_text(encoding="utf-8"))
        v = str((raw or {}).get("model", "") or "")
    except Exception:  # noqa: BLE001
        v = ""
    return v or DEFAULT_MODEL


def set_selected_model(name: str) -> bool:
    name = str(name or "").strip()
    if not name:
        return False
    try:
        with _LOCK:
            _PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _PREF_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": name}), encoding="utf-8")
            tmp.replace(_PREF_PATH)
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class ModelState:
    state: str = STATE_OFFLINE
    model: str = ""
    installed: List[str] = field(default_factory=list)
    detail: str = ""
    server_version: str = ""

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    def headline(self) -> str:
        """One sentence, naming the next action. A status that does not tell you
        what to do next is decoration."""
        if self.state == STATE_READY:
            return f"{self.model} ready"
        if self.state == STATE_NO_MODEL:
            return f"{self.model} is not downloaded on this machine"
        if self.state == STATE_ERROR:
            return f"the local model server returned an error: {self.detail}"
        return "the local model server is not running"

    def action(self) -> str:
        return {STATE_READY: "", STATE_NO_MODEL: "download",
                STATE_ERROR: "restart", STATE_OFFLINE: "start"}.get(self.state, "")


def _server_listening(timeout: float = _CONNECT_TIMEOUT) -> bool:
    """Is anything accepting connections at `base_url()`? Cheap and bounded.

    A short-circuit in front of the HTTP calls. "Nothing is listening" is the
    common case on any machine that does not run a local model, and answering it
    with a TCP connect costs milliseconds instead of the multi-second accumulation
    described on `_CONNECT_TIMEOUT`.

    Returns True on the first address that accepts. Any resolution or connection
    failure is False — this decides whether to bother, never whether a result is
    valid.
    """
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(base_url())
    host = parts.hostname or "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addrs = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for family, socktype, proto, _canon, sockaddr in addrs:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(timeout)
                sock.connect(sockaddr)
                return True
        except OSError:
            continue
    return False


def _get(path: str, timeout: float = _PROBE_TIMEOUT):
    req = urllib.request.Request(base_url() + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def installed_models(timeout: float = _PROBE_TIMEOUT) -> Optional[List[str]]:
    """Model names on this machine, or None when the server is unreachable.

    None and [] are different answers: None is "cannot tell", [] is "the server
    is up and has nothing", and only the second is the user's problem to fix.
    """
    try:
        data = _get("/api/tags", timeout)
    except Exception:  # noqa: BLE001
        return None
    out = []
    for m in (data.get("models") or []):
        name = str(m.get("name") or m.get("model") or "").strip()
        if name:
            out.append(name)
    return sorted(out)


def show(model: str = "", timeout: float = 6.0) -> Optional[dict]:
    """The server's own description of a model, or None when it cannot be had.

    This is the input to model morphometry: the architecture fields and, on a
    server new enough to publish it, the tensor inventory. Returned raw and
    unmerged — interpreting it belongs to
    `alelyon.runtime.vector.lattice.morphometry`, which is pure and testable
    without a server.

    None means "cannot tell", never "the model has no structure". The gate below
    is the same one `probe()` uses: on a machine with no server, two HTTP
    requests to `localhost` cost seconds, and this is called from a view.

    The timeout is longer than `_PROBE_TIMEOUT` on purpose — a show for a 30B
    model serialises a few thousand tensor records, which is more work than a
    version string — but it is still bounded, and callers run it off the UI
    thread.
    """
    model = str(model or "").strip() or selected_model()
    if not _server_listening():
        return None
    body = json.dumps({"model": model}).encode("utf-8")
    req = urllib.request.Request(base_url() + "/api/show", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - a missing description is an ordinary state
        return None
    return data if isinstance(data, dict) else None


def _matches(installed: List[str], want: str) -> bool:
    """Ollama reports `qwen3-coder:30b`; a user may have configured
    `qwen3-coder`. Treat a missing tag as :latest, the way the CLI does."""
    want = want.strip()
    if want in installed:
        return True
    if ":" not in want:
        return f"{want}:latest" in installed
    return False


def probe() -> ModelState:
    """Current state. Never raises, never blocks longer than the timeout.

    Refuses before the HTTP layer when nothing is listening. Without that gate
    the two requests below cost 8 seconds on a machine with no local model
    server — see `_CONNECT_TIMEOUT`.
    """
    want = selected_model()
    if not _server_listening():
        return ModelState(STATE_OFFLINE, want, [],
                          "nothing is listening at " + base_url())
    version = ""
    try:
        version = str((_get("/api/version") or {}).get("version", "") or "")
    except Exception:  # noqa: BLE001
        pass
    names = installed_models()
    if names is None:
        return ModelState(STATE_OFFLINE, want, [], "no response from " + base_url())
    if _matches(names, want):
        return ModelState(STATE_READY, want, names, "", version)
    return ModelState(STATE_NO_MODEL, want, names,
                      f"{len(names)} other model(s) installed", version)


# ── starting the server ──────────────────────────────────────────────────────
def ollama_binary() -> Optional[str]:
    """Ollama's own installer amends PATH for *new* shells only, so a GUI already
    running when the user installed it -- the exact moment they then press
    'Start' -- cannot see it by name. Resolve against the disk instead."""
    return toolpath.which("ollama")


def is_installed() -> bool:
    return ollama_binary() is not None


def ensure_running(wait: float = _START_GRACE) -> ModelState:
    """Start the server if it is installed and not listening, then re-probe.

    Only ever starts a server that is already on the machine. If Ollama is not
    installed this says so and stops — silently downloading a runtime would be a
    far larger act than the user asked for by clicking 'Start'.
    """
    st = probe()
    if st.state != STATE_OFFLINE:
        return st
    exe = ollama_binary()
    if exe is None:
        return ModelState(STATE_OFFLINE, st.model, [],
                          "Ollama could not be found on this machine. Install "
                          "it from ollama.com, then press Start again. "
                          + toolpath.find("ollama").reason())
    try:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            # Detached, no console window: the server outlives this GUI, which
            # is the point — restarting the app must not kill the model.
            kwargs["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                       | subprocess.DETACHED_PROCESS)
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return ModelState(STATE_ERROR, st.model, [],
                          f"could not start Ollama: {exc}")

    deadline = time.time() + max(1.0, float(wait))
    while time.time() < deadline:
        time.sleep(0.5)
        st = probe()
        if st.state != STATE_OFFLINE:
            return st
    return ModelState(STATE_OFFLINE, st.model, [],
                      f"started Ollama but it did not begin listening within "
                      f"{wait:.0f}s")


# ── pulling a model ──────────────────────────────────────────────────────────
def pull(model: str, on_progress: Optional[Callable[[str, float], None]] = None,
         should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """Download a model, streaming progress. Returns (ok, message).

    `on_progress(status, fraction)` is called as the server reports; fraction is
    -1.0 when the server gives a status with no byte counts, which is most of
    the run. Reporting -1 rather than 0 keeps a progress bar from sitting at
    zero through a ten-minute download and looking stalled.
    """
    model = str(model or "").strip() or selected_model()
    body = json.dumps({"model": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(base_url() + "/api/pull", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                if should_stop is not None and should_stop():
                    return False, "cancelled"
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if msg.get("error"):
                    return False, str(msg["error"])
                status = str(msg.get("status", "") or "")
                total = float(msg.get("total") or 0.0)
                done = float(msg.get("completed") or 0.0)
                frac = (done / total) if total > 0 else -1.0
                if on_progress is not None:
                    on_progress(status, frac)
                if status.lower() == "success":
                    return True, f"{model} downloaded"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    # The stream ended without a success line — check rather than assume.
    names = installed_models()
    if names is not None and _matches(names, model):
        return True, f"{model} downloaded"
    return False, "the download ended without confirming success"
