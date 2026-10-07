"""The platform's own llama.cpp server: one per model, loopback, a token per launch.

ADR-0041 (docs/architecture/adr/ADR-0041-local-inference-without-ollama.md)
replaces Ollama with llama.cpp's `llama-server`, run and owned by the platform:

* **The binary** comes from `ALELYON_LLAMA_SERVER` or from the pinned install in
  `~/.alelyon/llama`, recorded with its SHA-256 by `install`. It is never taken
  from an Ollama install.
* **Models are GGUF files** in `~/.alelyon/models` (`ALELYON_MODELS_DIR`
  overrides). Listing is a scan; self-description is the file's own header
  (`gguf_header`). Nothing is pulled.
* **The server** binds 127.0.0.1 on a port chosen at launch. It takes its API key
  from a file holding a random token made for that launch (`--api-key-file`), so
  the token is not on a command line that other processes can read. It runs with
  `--offline` and no web UI, and inside a Windows Job Object
  (`process_custody`), so it cannot outlive the process that owns it.
* **One model at a time**: asking for another model restarts the server. An idle
  server is stopped to give its VRAM back (Angel shares the RX).
* **A grammar is measured, not assumed.** `probe_grammar` asks for a
  schema-shaped answer to a prompt that invites prose. A provider claims a
  grammar only after the probe recorded success for this binary and model.

Everything that reaches the network here reaches 127.0.0.1 only.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from alelyon.runtime.oracle.assistant import gguf_header

BINARY_ENV = "ALELYON_LLAMA_SERVER"
MODELS_ENV = "ALELYON_MODELS_DIR"
LOOPBACK = "127.0.0.1"
BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"

DEFAULT_CTX = 8192
DEFAULT_GPU_LAYERS = 999          # every layer that fits; llama.cpp clamps
DEFAULT_PARALLEL = 1
DEFAULT_IDLE_SECONDS = 600.0
START_TIMEOUT_S = 180.0           # a 20 GB model takes a while to map and upload
HEALTH_TIMEOUT_S = 2.0
STOP_TIMEOUT_S = 10.0

#: llama.cpp reads every `LLAMA_ARG_*` variable as a default for a flag, so a
#: stray one (an earlier tool exported some) silently changes this launch.
_STRIPPED_ENV_PREFIXES = ("LLAMA_ARG_", "OLLAMA_")


#: Every request here goes to 127.0.0.1, and goes there DIRECTLY: an inherited
#: `HTTP_PROXY` must never see a local request or its launch token.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LlamaServerError(RuntimeError):
    """The managed server could not be found, started or used."""


class ModelNotFound(LlamaServerError):
    """A model name that maps to no GGUF file. Never guessed."""


# ── where things are ─────────────────────────────────────────────────────────

def alelyon_home() -> Path:
    return Path.home() / ".alelyon"


def models_dir() -> Path:
    override = os.environ.get(MODELS_ENV, "").strip()
    return Path(override) if override else alelyon_home() / "models"


def llama_dir() -> Path:
    return alelyon_home() / "llama"


def _settings_path() -> Path:
    return llama_dir() / "settings.json"


def _probe_record_path() -> Path:
    return llama_dir() / "grammar-probes.json"


def _is_ollama_path(path: Path) -> bool:
    return "ollama" in str(path).lower()


def find_binary() -> Optional[Path]:
    """The llama-server to run, or None. Never an Ollama install's copy."""
    override = os.environ.get(BINARY_ENV, "").strip()
    candidates = [Path(override)] if override else [llama_dir() / BINARY_NAME]
    for candidate in candidates:
        if candidate.is_file() and not _is_ollama_path(candidate):
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install(source_dir: str | Path) -> Dict[str, Any]:
    """Copy a llama.cpp build folder into `~/.alelyon/llama`, pinned by SHA-256.

    The source must hold `llama-server` and the libraries beside it. Files are
    copied, never moved, and the manifest records each file's SHA-256 so a later
    reader can tell which build is installed.
    """
    source = Path(source_dir)
    if _is_ollama_path(source):
        raise LlamaServerError("refusing a llama.cpp build from an Ollama install")
    if not (source / BINARY_NAME).is_file():
        raise LlamaServerError(f"no {BINARY_NAME} in {source}")
    target = llama_dir()
    target.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    for item in sorted(source.iterdir()):
        if item.is_file() and item.suffix.lower() in {".exe", ".dll", ".so", ".dylib", ""}:
            shutil.copy2(item, target / item.name)
            files[item.name] = sha256_file(target / item.name)
    manifest = {"source": str(source), "installed_at": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    return manifest


# ── models are files ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LocalModel:
    name: str          # the file name without `.gguf`; what a user picks
    path: Path
    size: int


def list_models(root: Optional[Path] = None) -> List[LocalModel]:
    """GGUF models in the models folder and one level below it.

    Vision projectors (`mmproj*`) are companions, not models, and are left out.
    """
    base = Path(root) if root is not None else models_dir()
    if not base.is_dir():
        return []
    found: Dict[str, LocalModel] = {}
    for pattern in ("*.gguf", "*/*.gguf"):
        for path in sorted(base.glob(pattern)):
            if path.name.lower().startswith("mmproj") or not path.is_file():
                continue
            if not gguf_header.is_gguf(path):
                continue
            name = path.stem
            if name not in found:
                found[name] = LocalModel(name, path, path.stat().st_size)
    return sorted(found.values(), key=lambda m: m.name.lower())


def resolve_model(name: str, root: Optional[Path] = None) -> LocalModel:
    """The model a name means: its file stem or file name, exactly."""
    wanted = str(name or "").strip()
    if not wanted:
        raise ModelNotFound("no model is selected")
    stem = wanted[:-5] if wanted.lower().endswith(".gguf") else wanted
    for model in list_models(root):
        if model.name == stem:
            return model
    raise ModelNotFound(f"no GGUF file named {stem!r} in {Path(root) if root else models_dir()}")


def describe_model(name: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """What `/api/show` used to say, read from the file's own header."""
    model = resolve_model(name, root)
    facts = gguf_header.read_header(model.path).describe()
    facts.update({"model": model.name, "path": str(model.path), "size": model.size})
    return facts


# ── settings ─────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """How the server runs. Which model it runs is the analyst's model
    preference (`local_model.selected_model`), not a setting of the server."""
    ctx_size: int = DEFAULT_CTX
    gpu_layers: int = DEFAULT_GPU_LAYERS
    parallel: int = DEFAULT_PARALLEL
    idle_seconds: float = DEFAULT_IDLE_SECONDS


def load_settings() -> Settings:
    try:
        raw = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    settings = Settings()
    for key in ("ctx_size", "gpu_layers", "parallel"):
        if isinstance(raw.get(key), int) and raw[key] >= 0:
            setattr(settings, key, raw[key])
    if isinstance(raw.get("idle_seconds"), (int, float)) and raw["idle_seconds"] > 0:
        settings.idle_seconds = float(raw["idle_seconds"])
    return settings


def save_settings(settings: Settings) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(settings.__dict__, indent=2), encoding="utf-8")
    temp.replace(path)


def selected_model() -> str:
    """The model Local uses (`local_model.selected_model`)."""
    # local_model imports this module, so it is imported here, at call time.
    from alelyon.runtime.oracle.assistant import local_model  # noqa: PLC0415

    return local_model.selected_model()


def select_model(name: str) -> LocalModel:
    """Remember the model Local uses. Refuses a name that maps to no file."""
    model = resolve_model(name)
    from alelyon.runtime.oracle.assistant import local_model  # noqa: PLC0415

    if not local_model.record_selected_model(model.name):
        raise LlamaServerError(f"could not record {model.name!r} as the selected model")
    return model


# ── the server ───────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def _clean_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    source = dict(os.environ if environ is None else environ)
    return {k: v for k, v in source.items()
            if not k.upper().startswith(_STRIPPED_ENV_PREFIXES)}


def _default_launcher(args: List[str], *, env: Dict[str, str], log: Path):
    """Start the server, inside a kill-on-close Job Object where Windows allows.

    Returns `(process, custody)`; `custody` is None when only a plain child
    process could be made, which is said in `ManagedServer.custody`.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # The child inherits its own copy of the log handle; this process's copy is
    # closed as soon as the child exists, so nothing here keeps the file open.
    with open(log, "ab") as handle:
        kwargs = dict(env=env, stdin=subprocess.DEVNULL, stdout=handle,
                      stderr=subprocess.STDOUT)
        if os.name == "nt":
            try:
                from alelyon.runtime.common import process_custody as PC  # noqa: PLC0415
                custody = PC.spawn(args, creationflags=flags, **kwargs)
                return custody.process, custody
            except Exception:  # noqa: BLE001 - fall back, and say so
                pass
        return subprocess.Popen(args, creationflags=flags, **kwargs), None


@dataclass
class ManagedServer:
    """One llama-server process serving one model."""

    binary: Path
    model: LocalModel
    ctx_size: int = DEFAULT_CTX
    gpu_layers: int = DEFAULT_GPU_LAYERS
    parallel: int = DEFAULT_PARALLEL
    launcher: Callable[..., Any] = field(default=_default_launcher, repr=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    port: int = 0
    token: str = field(default="", repr=False)
    process: Any = field(default=None, repr=False)
    custody: Any = field(default=None, repr=False)
    last_used: float = 0.0
    #: Requests in flight. An idle stop never interrupts one.
    inflight: int = 0
    _key_dir: Optional[Path] = field(default=None, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    @property
    def api_base(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def alias(self) -> str:
        return self.model.name

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def command(self, key_file: Path, log: Path) -> List[str]:
        return [str(self.binary),
                "--host", LOOPBACK, "--port", str(self.port),
                "-m", str(self.model.path), "--alias", self.alias,
                "-c", str(self.ctx_size), "-ngl", str(self.gpu_layers),
                "-np", str(self.parallel),
                "--api-key-file", str(key_file),
                "--no-webui", "--offline"]

    def start(self, timeout_s: float = START_TIMEOUT_S) -> None:
        if self.running:
            return
        self.port = _free_port()
        self.token = secrets.token_urlsafe(32)
        self._key_dir = Path(tempfile.mkdtemp(prefix="alelyon-llama-"))
        key_file = self._key_dir / "api-key"
        key_file.write_text(self.token + "\n", encoding="utf-8")
        log = llama_dir() / "logs" / f"{self.model.name}.log"
        self.process, self.custody = self.launcher(
            self.command(key_file, log), env=_clean_env(), log=log)
        deadline = self.clock() + timeout_s
        while self.clock() < deadline:
            if not self.running:
                self._forget_key()
                raise LlamaServerError(
                    f"llama-server exited while loading {self.model.name} "
                    f"(code {self.process.poll()}); see {log}")
            if self.healthy():
                self.touch()
                return
            time.sleep(0.25)
        self.stop()
        raise LlamaServerError(f"llama-server did not become ready in {timeout_s:.0f} s")

    def healthy(self) -> bool:
        try:
            with _DIRECT.open(f"{self.base_url}/health",
                              timeout=HEALTH_TIMEOUT_S) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def touch(self) -> None:
        self.last_used = self.clock()

    def begin(self) -> None:
        with self._lock:
            self.inflight += 1
            self.touch()

    def end(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self.touch()

    def idle_for(self) -> float:
        return self.clock() - self.last_used

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"}

    def stop(self) -> None:
        process, custody = self.process, self.custody
        self.process = self.custody = None
        try:
            if custody is not None:
                custody.close(timeout_s=STOP_TIMEOUT_S)
            elif process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            self._forget_key()

    def _forget_key(self) -> None:
        if self._key_dir is not None:
            shutil.rmtree(self._key_dir, ignore_errors=True)
            self._key_dir = None


class Manager:
    """The one server this process runs: started on demand, stopped when idle."""

    def __init__(self, *, binary_finder: Callable[[], Optional[Path]] = find_binary,
                 settings_loader: Callable[[], Settings] = load_settings,
                 launcher: Callable[..., Any] = _default_launcher,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._binary_finder = binary_finder
        self._settings_loader = settings_loader
        self._launcher = launcher
        self._clock = clock
        self._lock = threading.RLock()
        self._server: Optional[ManagedServer] = None
        self._watcher: Optional[threading.Thread] = None

    @property
    def current(self) -> Optional[ManagedServer]:
        return self._server

    def ensure(self, model_name: str = "") -> ManagedServer:
        """A running server for `model_name` (or the selected model)."""
        with self._lock:
            settings = self._settings_loader()
            model = resolve_model(model_name or selected_model())
            server = self._server
            if server is not None and server.model.path == model.path and server.running:
                server.touch()
                return server
            if server is not None:
                server.stop()
                self._server = None
            binary = self._binary_finder()
            if binary is None:
                raise LlamaServerError(
                    f"no llama-server: set {BINARY_ENV} or install a llama.cpp "
                    f"build into {llama_dir()}")
            server = ManagedServer(binary=binary, model=model,
                                   ctx_size=settings.ctx_size,
                                   gpu_layers=settings.gpu_layers,
                                   parallel=settings.parallel,
                                   launcher=self._launcher, clock=self._clock)
            server.start()
            self._server = server
            self._watch(settings.idle_seconds)
            return server

    def stop(self) -> None:
        with self._lock:
            if self._server is not None:
                self._server.stop()
                self._server = None

    def stop_if_idle(self, idle_seconds: float) -> bool:
        with self._lock:
            server = self._server
            if (server is not None and server.inflight == 0
                    and server.idle_for() >= idle_seconds):
                self.stop()
                return True
            return False

    def _watch(self, idle_seconds: float) -> None:
        if self._watcher is not None and self._watcher.is_alive():
            return

        def loop() -> None:
            while self._server is not None:
                time.sleep(min(30.0, max(1.0, idle_seconds / 10)))
                self.stop_if_idle(idle_seconds)

        self._watcher = threading.Thread(target=loop, name="llama-server-idle",
                                         daemon=True)
        self._watcher.start()


_MANAGER: Optional[Manager] = None
_MANAGER_LOCK = threading.Lock()


def manager() -> Manager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = Manager()
            atexit.register(_MANAGER.stop)
        return _MANAGER


# ── requests ─────────────────────────────────────────────────────────────────

def post_json(server: ManagedServer, path: str, payload: Dict[str, Any], *,
              timeout: float = 120.0) -> Dict[str, Any]:
    """POST to the server with its launch token. Loopback only, by construction."""
    request = urllib.request.Request(
        f"{server.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
        headers=server.headers(), method="POST")
    server.begin()
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    finally:
        server.end()


def response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The OpenAI-style request llama-server compiles into a sampler grammar."""
    return {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}


#: Thinking models (Qwen3 and others) reason before they answer, and llama.cpp
#: applies the schema's grammar to the answer only. A constrained request turns
#: thinking off so the whole token budget goes to the constrained answer.
#: Templates without the switch ignore it.
NO_THINKING = {"enable_thinking": False}


def constrained_payload(model: str, prompt: str, schema: Dict[str, Any], *,
                        max_tokens: Optional[int] = None,
                        temperature: float = 0.0) -> Dict[str, Any]:
    """A chat request whose answer llama-server restricts to `schema`."""
    payload: Dict[str, Any] = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "stream": False,
        "response_format": response_format(schema),
        "chat_template_kwargs": dict(NO_THINKING),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max(1, int(max_tokens))
    return payload


# ── the grammar probe ────────────────────────────────────────────────────────

PROBE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
PROBE_PROMPT = ("Write four lines of free verse about the sea. Do not use JSON, "
                "braces or numbers.")


def conforms_to_probe(text: str) -> bool:
    """Is `text` exactly an object with one integer `answer`, as the schema asks?"""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False
    return (isinstance(value, dict) and set(value) == {"answer"}
            and isinstance(value["answer"], int)
            and not isinstance(value["answer"], bool))


def probe_grammar(server: ManagedServer, *, timeout: float = 120.0) -> bool:
    """Does this server, on this model, restrict sampling to a JSON Schema?

    The prompt asks for prose. Only a sampler restricted by the schema answers
    it with `{"answer": <int>}`. A server that treats `response_format` as a
    suggestion answers with a poem, and the probe says no.

    Thinking is switched off for the probe, as it is for every constrained
    request (`constrained_payload`). MEASURED 2026-10-02 on llama.cpp 0.4.1-dev
    (fb27a52) with Qwen3-0.6B: with thinking on, the 64-token budget went to the
    reasoning and the grammar never reached the answer (`finish=length`, empty
    content); with thinking off, the schema gave `{ "answer": 4 }` and the same
    prompt without it gave prose.
    """
    payload = constrained_payload(server.alias, PROBE_PROMPT, PROBE_SCHEMA,
                                  max_tokens=64)
    try:
        body = post_json(server, "/v1/chat/completions", payload, timeout=timeout)
        text = body["choices"][0]["message"]["content"]
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError):
        return False
    return conforms_to_probe(text)


def _probe_key(server: ManagedServer) -> str:
    stat = server.model.path.stat()
    return f"{sha256_file(server.binary)}|{server.model.path}|{stat.st_size}|{int(stat.st_mtime)}"


def recorded_grammar(model_name: str = "") -> bool:
    """Has a probe recorded a working grammar for the installed binary and this model?"""
    binary = find_binary()
    try:
        model = resolve_model(model_name or selected_model())
    except ModelNotFound:
        return False
    if binary is None:
        return False
    try:
        records = json.loads(_probe_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    probe = ManagedServer(binary=binary, model=model)
    return records.get(_probe_key(probe)) is True if isinstance(records, dict) else False


def record_grammar_probe(server: ManagedServer) -> bool:
    """Run the probe on a running server and remember the answer for this pair."""
    result = probe_grammar(server)
    path = _probe_record_path()
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, dict):
            records = {}
    except (OSError, ValueError):
        records = {}
    records[_probe_key(server)] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return result


# ── command line ─────────────────────────────────────────────────────────────

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m alelyon.runtime.oracle.assistant.llama_server",
        description="The platform's llama.cpp server (ADR-0041).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list GGUF models and what their headers say")
    inst = sub.add_parser("install", help="pin a llama.cpp build into ~/.alelyon/llama")
    inst.add_argument("source")
    sel = sub.add_parser("select", help="choose the model Local uses")
    sel.add_argument("model")
    probe = sub.add_parser("probe", help="start the server and measure its grammar")
    probe.add_argument("model", nargs="?", default="")
    args = parser.parse_args(argv)

    if args.command == "models":
        models = list_models()
        if not models:
            print(f"no GGUF models in {models_dir()}")
        for model in models:
            facts = gguf_header.read_header(model.path).describe()
            print(f"{model.name:48} {model.size / 2**30:6.2f} GiB  "
                  f"{facts['architecture']:10} {facts['quantization']:8} "
                  f"ctx {facts['context_length']}  params {facts['parameters']:,}")
        return 0
    if args.command == "install":
        manifest = install(args.source)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "select":
        print(f"selected {select_model(args.model).name}")
        return 0
    if args.command == "probe":
        server = manager().ensure(args.model)
        try:
            ok = record_grammar_probe(server)
            print(f"{server.model.name}: grammar {'ENFORCED' if ok else 'NOT enforced'}")
            return 0 if ok else 1
        finally:
            manager().stop()
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BINARY_ENV", "LlamaServerError", "LocalModel", "Manager", "ManagedServer",
    "MODELS_ENV", "ModelNotFound", "NO_THINKING", "PROBE_SCHEMA", "Settings",
    "conforms_to_probe", "constrained_payload",
    "describe_model", "find_binary", "install", "list_models", "llama_dir",
    "load_settings", "manager", "models_dir", "post_json", "probe_grammar",
    "record_grammar_probe", "recorded_grammar", "resolve_model", "response_format",
    "save_settings", "select_model", "selected_model",
]
