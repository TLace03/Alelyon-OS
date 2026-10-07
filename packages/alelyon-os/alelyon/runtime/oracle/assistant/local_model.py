"""The local model, managed from inside the app.

Since ADR-0041 the local model is the platform's own llama.cpp server
(`llama_server`), not Ollama. This module keeps the small surface the app was
built around (probe, list, show, select, start) so the model bar, the Lattice
panels and the Lattice service ask the same questions they always asked; the
answers now come from files and from the managed server.

**Three states, three different sentences.** Collapsing them is how "the
analyst is broken" gets reported for unrelated causes:

    OFFLINE     no llama-server is installed            → install a llama.cpp build
    NO_MODEL    the selected model is no file here      → put a GGUF in the folder
    READY       a server binary and the model's file    → ask away
    ERROR       the server failed to start or answer    → retry, and read why

READY does not mean a server is running. The server starts on the first
question and stops when idle, so READY means "a question will be answered".

**Nothing here downloads anything.** A model is a GGUF file someone puts in
`~/.alelyon/models`; fetching weights is an explicit, owner-approved act
(ADR-0041), never a button that starts a transfer.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from alelyon.runtime.oracle.assistant import gguf_header
from alelyon.runtime.oracle.assistant import llama_server as LS

#: There is no default model name: a model is a file, and naming one that is not
#: on the machine is how a fresh install used to point at a 17 GB download.
DEFAULT_MODEL = ""
MAX_MODEL_NAME_CHARS = 4_096
MAX_MODEL_PREF_BYTES = 8_192

#: Where a saved Ollama row with no URL of its own pointed when `OLLAMA_BASE_URL`
#: was unset. Ollama is retired (ADR-0041) and nothing here calls it.
OLLAMA_DEFAULT_BASE = "http://localhost:11434"

#: The selected model is runtime state, not a GUI preference: the API server,
#: any headless caller and the native Lattice (`lattice-core/src/local_model.rs`)
#: read it too, so it lives in `<globals>/analyst_model.json`. ``None`` keeps
#: Runtime paths unbound until the preference is actually requested; tests and
#: `tools/lattice_native_parity.py` point it at a temporary file.
_PREF_PATH: Optional[Path] = None

STATE_OFFLINE = "offline"
STATE_NO_MODEL = "no_model"
STATE_READY = "ready"
STATE_ERROR = "error"

#: Worst-case wall time of one `probe()`: a folder scan and a few `stat` calls,
#: no network. A caller that joins a probe running on a thread uses this bound;
#: a `QThread` still running when it is destroyed aborts the process, so the
#: bound is generous rather than tight.
PROBE_WORST_CASE = 2.0

_LOCK = threading.Lock()


def _normalise_model_name(value: object) -> str:
    """Return one bounded UTF-8 model identifier, or an empty refusal."""
    if type(value) is not str:
        return ""
    value = value.strip()
    if not value or len(value) > MAX_MODEL_NAME_CHARS:
        return ""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    return value


def base_url() -> str:
    """The address a saved Ollama row with no URL of its own would have called:
    `OLLAMA_BASE_URL` (trailing `/` removed), else the default loopback daemon.

    Ollama is retired (ADR-0041) and nothing calls this address. It is read only
    to judge such a row's locality (`model_config.ollama_is_local`), so a row
    that pointed off this machine is never reported as local. As before, the
    variable is not trimmed: spaces are an address no request can use.
    """
    return (os.environ.get("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE).rstrip("/")


def _pref_path() -> Path:
    if _PREF_PATH is not None:
        return Path(_PREF_PATH)
    from alelyon.runtime.common.paths import GLOBALS_DIR  # noqa: PLC0415

    return Path(GLOBALS_DIR) / "analyst_model.json"


def selected_model() -> str:
    """The model Local uses: a GGUF file name in the models folder, or "".

    Read from `analyst_model.json`, at most `MAX_MODEL_PREF_BYTES` of it; a file
    that is larger, unreadable or malformed means "no preference". There is no
    default and no environment override: `OLLAMA_MODEL` named an Ollama tag, and
    a default named a download (ADR-0041). A name left from Ollama is returned
    as it is, so the model bar can say it is not a file here.
    """
    try:
        with _pref_path().open("rb") as source:
            encoded = source.read(MAX_MODEL_PREF_BYTES + 1)
        if len(encoded) > MAX_MODEL_PREF_BYTES:
            return DEFAULT_MODEL
        raw = json.loads(encoded)
        name = _normalise_model_name(raw.get("model", "") if type(raw) is dict else "")
    except Exception:  # noqa: BLE001 - an unreadable preference is no preference
        name = ""
    return name or DEFAULT_MODEL


def record_selected_model(name: str) -> bool:
    """Persist `name` as the selected model without checking for its file.
    False when it cannot be persisted. `set_selected_model` is the checked
    entry point; `llama_server.select_model` calls this after resolving."""
    name = _normalise_model_name(name)
    if not name:
        return False
    try:
        payload = json.dumps({"model": name}, ensure_ascii=False).encode("utf-8")
    except (TypeError, UnicodeEncodeError):
        return False
    # Anything accepted for persistence must be readable through the exact
    # bounded reader `selected_model` uses on the next call.
    if len(payload) > MAX_MODEL_PREF_BYTES:
        return False
    try:
        with _LOCK:
            pref = _pref_path()
            pref.parent.mkdir(parents=True, exist_ok=True)
            tmp = pref.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(pref)
        return True
    except Exception:  # noqa: BLE001 - a preference write never breaks the panel
        return False


def set_selected_model(name: str) -> bool:
    """Choose the model Local uses. Refuses a name that is no file here."""
    name = _normalise_model_name(name)
    if not name:
        return False
    try:
        found = LS.resolve_model(name)
    except Exception:  # noqa: BLE001 - LS.ModelNotFound, or a folder that cannot be read
        return False
    return record_selected_model(found.name)


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
            if self.model:
                return f"{self.model} is not in {LS.models_dir()}"
            return f"no model chosen: put a GGUF file in {LS.models_dir()}"
        if self.state == STATE_ERROR:
            return f"the local model server failed: {self.detail}"
        return "llama.cpp's server is not installed on this machine"

    def action(self) -> str:
        """The one button the model bar may offer. Only a failed server has one:
        a model is chosen in the list and installed by putting a file in place."""
        return "restart" if self.state == STATE_ERROR else ""


def installed_models(timeout: float = 0.0) -> Optional[List[str]]:
    """GGUF models in the models folder. Never None: the folder always answers.

    `timeout` is accepted for the callers that pass one; a scan does not wait.
    """
    del timeout
    try:
        return [model.name for model in LS.list_models()]
    except OSError:
        return None


def show(model: str = "", timeout: float = 6.0) -> Optional[dict]:
    """The model's description, read from its GGUF header, in the shape
    `/api/show` had: `model`, `details`, `model_info` and `tensors`.

    None means "cannot tell" (no such file, or a header this reader refuses),
    never "the model has no structure".
    """
    del timeout
    name = str(model or "").strip() or selected_model()
    try:
        found = LS.resolve_model(name)
        return gguf_header.read_header(found.path).show_payload(found.name)
    except (LS.LlamaServerError, gguf_header.GGUFError, OSError):
        return None


def server_binary() -> Optional[str]:
    """The llama-server this machine would run, or None."""
    binary = LS.find_binary()
    return str(binary) if binary is not None else None


def is_installed() -> bool:
    return server_binary() is not None


def probe() -> ModelState:
    """Current state. Never raises; touches the disk only, never the network."""
    want = selected_model()
    try:
        names = installed_models() or []
    except Exception:  # noqa: BLE001
        names = []
    if not is_installed():
        return ModelState(STATE_OFFLINE, want, names,
                          f"no llama-server: set {LS.BINARY_ENV} or install a "
                          f"llama.cpp build into {LS.llama_dir()}")
    current = LS.manager().current
    if want and want in names:
        version = "running" if current is not None and current.running else ""
        return ModelState(STATE_READY, want, names, "", version)
    detail = (f"{len(names)} model(s) in {LS.models_dir()}" if names
              else f"no GGUF files in {LS.models_dir()}")
    return ModelState(STATE_NO_MODEL, want, names, detail)


def ensure_running(wait: float = LS.START_TIMEOUT_S) -> ModelState:
    """Start (or restart) the server for the selected model, then re-probe."""
    st = probe()
    if st.state in (STATE_OFFLINE, STATE_NO_MODEL):
        return st
    try:
        LS.manager().ensure(st.model)
    except LS.LlamaServerError as exc:
        return ModelState(STATE_ERROR, st.model, st.installed, str(exc))
    return probe()


def pull(model: str, on_progress: Optional[Callable[[str, float], None]] = None,
         should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """Refused: models are files (ADR-0041). Returns (False, what to do instead)."""
    del model, on_progress, should_stop
    return False, (f"models are files now: put a GGUF file in {LS.models_dir()} "
                   "and choose it in the list")


__all__ = [
    "DEFAULT_MODEL", "MAX_MODEL_NAME_CHARS", "MAX_MODEL_PREF_BYTES", "ModelState",
    "OLLAMA_DEFAULT_BASE", "PROBE_WORST_CASE", "STATE_ERROR", "STATE_NO_MODEL",
    "STATE_OFFLINE", "STATE_READY", "base_url", "ensure_running",
    "installed_models", "is_installed", "probe", "pull", "record_selected_model",
    "selected_model", "server_binary", "set_selected_model", "show",
]
