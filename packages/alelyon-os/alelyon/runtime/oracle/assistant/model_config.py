"""Which models this machine can reach, as declared configuration.

Before this, "which model answers" was two hard-coded possibilities: a local
Ollama server, and Anthropic when `ANTHROPIC_API_KEY` happened to be set. That
is fine for one developer and wrong for anyone else — it cannot express "my own
fine-tuned weights, served by vLLM on this box", and it cannot express "my
employer's OpenAI key", which are the two things a user actually has.

So endpoints become data. An entry is:

    id, label, kind, base_url, model, api_key_name

and that single shape covers every case, because `/v1/chat/completions` is the
de-facto wire format: vLLM, TGI, llama.cpp, LM Studio, OpenAI, Groq, Together,
Fireworks, DeepSeek, Mistral, xAI and OpenRouter all speak it. Custom weights and
a frontier lab differ by URL and whether a key is attached — not by code path.

Two rules this module exists to enforce
---------------------------------------
**Key VALUES are never stored here.** An entry holds an api_key_name; the value
is resolved through `atlas.data.keys.get_key()` at call time. This file is
JSON in the state root — the kind of thing a user attaches to a bug report — and
a secret written into it is a secret leaked by the next person trying to be
helpful (AGENTS.md §9).

**"Local" is decided by the URL, never by the label.** `Chain.mark_private()`
stops desk and book context reaching a non-local model, and it trusts
`Provider.local`. If that flag came from a user-supplied name, anyone could label
`https://api.example.com` as "Local Qwen" and silently route positions off the
machine. `is_local_url()` resolves the host and admits loopback only.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

KIND_OLLAMA = "ollama"
KIND_OPENAI = "openai-compatible"
KIND_ANTHROPIC = "anthropic"
KINDS = (KIND_OLLAMA, KIND_OPENAI, KIND_ANTHROPIC)

#: An api_key_name must look like an environment variable, because that is what
#: it becomes. Refusing anything else keeps a crafted name out of the env file.
_KEY_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 1_048_576
MAX_ENDPOINT_ROWS = 10_000
MAX_ENDPOINT_FIELD_CHARS = 4_096
_NATIVE_PATH_TYPE = type(Path())


def config_path(*, state_root: Path | None = None) -> Path:
    """Resolve the registry path without freezing Runtime paths at import.

    Entry points may supply the pure deployment ``state_root`` to inspect the
    registry before bootstrapping or creating state. Existing callers that omit
    it retain the Runtime-path default, resolved lazily. The explicit model
    configuration override remains highest precedence.
    """
    override = (os.environ.get("ALELYON_MODEL_CONFIG") or "").strip()
    if override:
        return Path(override)
    if state_root is not None:
        if type(state_root) is not _NATIVE_PATH_TYPE:
            raise TypeError("state_root must be a platform Path")
        return state_root / "globals" / "model_endpoints.json"
    from alelyon.runtime.common.paths import GLOBALS_DIR

    return Path(GLOBALS_DIR) / "model_endpoints.json"


def __getattr__(name: str):
    """Keep the legacy ``CONFIG_PATH`` module attribute lazy."""
    if name == "CONFIG_PATH":
        return config_path()
    raise AttributeError(name)


def _bounded_text(value: object, *, nonempty: bool = False) -> bool:
    if type(value) is not str:
        return False
    if (nonempty and not value) or len(value) > MAX_ENDPOINT_FIELD_CHARS:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def is_local_url(url: str) -> bool:
    """Does this URL address THIS machine?

    Loopback only. A private-range address (192.168.x, 10.x) is another computer:
    it may be trusted, but it is not this one, and the private-context rule is
    about data leaving the machine that holds the book.

    Anything unparseable is not local. A hostname that is not literally loopback
    is not local either — resolving it here would mean a DNS lookup deciding a
    privacy boundary, and DNS is attacker-influenced.
    """
    try:
        host = (urlsplit(str(url or "").strip()).hostname or "").strip("[]")
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class ModelEndpoint:
    """One reachable model."""

    id: str
    label: str
    kind: str = KIND_OPENAI
    base_url: str = ""
    model: str = ""
    #: Name of the key, never its value.
    api_key_name: str = ""
    enabled: bool = True
    #: True for entries this module ships; a built-in may be edited but the flag
    #: lets the UI explain why an entry reappeared after being deleted.
    builtin: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not _bounded_text(self.id, nonempty=True):
            raise ValueError("endpoint id must be a bounded non-empty string")
        if not _bounded_text(self.label, nonempty=True):
            raise ValueError("endpoint label must be a bounded non-empty string")
        for value in (self.base_url, self.model, self.api_key_name, self.note):
            if not _bounded_text(value):
                raise ValueError("endpoint text fields must be bounded strings")
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError("endpoint kind is invalid")
        if type(self.enabled) is not bool or type(self.builtin) is not bool:
            raise TypeError("endpoint flags must be exact bool values")
        if self.api_key_name and not _KEY_NAME_RE.fullmatch(self.api_key_name):
            raise ValueError("endpoint key name is invalid")

    @property
    def local(self) -> bool:
        """Whether this endpoint is on this machine — decided by URL.

        Ollama with no explicit URL means the default loopback server.
        """
        if self.kind == KIND_ANTHROPIC:
            return False
        if self.kind == KIND_OLLAMA and not self.base_url:
            return True
        return is_local_url(self.base_url)

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_name)

    def has_key(self) -> bool:
        """Is the key this entry names actually present? Never raises."""
        if not self.api_key_name:
            return True
        try:
            from alelyon.runtime.atlas.data.keys import get_key
            return bool(get_key(self.api_key_name))
        except Exception:  # noqa: BLE001
            return False

    def ready(self) -> bool:
        return bool(self.enabled and self.model and self.has_key())

    def status(self) -> str:
        """One sentence a user can act on."""
        if not self.enabled:
            return "disabled"
        if not self.model:
            return "no model name set"
        if self.needs_key and not self.has_key():
            return f"needs {self.api_key_name}"
        return "ready"


class LoadIssue(str, Enum):
    """Closed, content-free reasons a registry observation is incomplete."""

    UNREADABLE = "unreadable"
    CORRUPT = "corrupt"
    UNSUPPORTED_ROOT = "unsupported-root"
    UNSUPPORTED_SCHEMA = "unsupported-schema"
    INVALID_ROW = "invalid-row"
    DUPLICATE_ENDPOINT_ID = "duplicate-endpoint-id"


@dataclass(frozen=True, slots=True)
class ModelConfigLoadReport:
    """A bounded registry observation without readiness or provider effects.

    Endpoint values remain available to the owning Oracle adapter, but are
    deliberately excluded from the report representation.  Reasons describe
    only structural failure classes and never reproduce source content or an
    exception raised while reading it.
    """

    endpoints: tuple[ModelEndpoint, ...] = field(repr=False)
    complete: bool
    reason_classes: tuple[LoadIssue, ...]

    def __post_init__(self) -> None:
        if type(self.endpoints) is not tuple or any(
            type(endpoint) is not ModelEndpoint for endpoint in self.endpoints
        ):
            raise TypeError("endpoints must be a tuple of exact ModelEndpoint values")
        if len(self.endpoints) > MAX_ENDPOINT_ROWS + len(_builtins()):
            raise ValueError("endpoint report exceeds its size bound")
        if type(self.complete) is not bool:
            raise TypeError("complete must be an exact bool")
        if type(self.reason_classes) is not tuple or any(
            type(reason) is not LoadIssue for reason in self.reason_classes
        ):
            raise TypeError("reason_classes must contain exact LoadIssue values")
        if len(set(self.reason_classes)) != len(self.reason_classes):
            raise ValueError("reason_classes must be unique")
        if self.complete is not (not self.reason_classes):
            raise ValueError("complete must agree with reason_classes")


# ── the built-in catalogue ───────────────────────────────────────────────────
#
# Base URLs are stable; MODEL NAMES ARE NOT — vendors rename and retire them.
# Every model field here is a starting point the user is expected to edit, which
# is why the UI shows it as a text field rather than a fixed label.

def _builtins() -> List[ModelEndpoint]:
    return [
        ModelEndpoint(
            id="ollama-local", label="Ollama (this machine)", kind=KIND_OLLAMA,
            base_url="", model="qwen3-coder:30b", builtin=True,
            note="Keyless. Runs entirely on this computer.",
        ),
        ModelEndpoint(
            id="local-openai", label="Local server (vLLM / TGI / llama.cpp / LM Studio)",
            kind=KIND_OPENAI, base_url="http://localhost:8000/v1", model="",
            enabled=False, builtin=True,
            note="Point this at your own served weights. Keyless by default; "
                 "set a key name if your server requires one.",
        ),
        ModelEndpoint(
            id="anthropic", label="Anthropic", kind=KIND_ANTHROPIC,
            model="claude-sonnet-4-5-20250929",
            api_key_name="ANTHROPIC_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="openai", label="OpenAI", kind=KIND_OPENAI,
            base_url="https://api.openai.com/v1", model="gpt-4o",
            api_key_name="OPENAI_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="google", label="Google Gemini", kind=KIND_OPENAI,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.0-flash", api_key_name="GEMINI_API_KEY",
            enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="xai", label="xAI Grok", kind=KIND_OPENAI,
            base_url="https://api.x.ai/v1", model="grok-2-latest",
            api_key_name="XAI_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="deepseek", label="DeepSeek", kind=KIND_OPENAI,
            base_url="https://api.deepseek.com/v1", model="deepseek-chat",
            api_key_name="DEEPSEEK_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="mistral", label="Mistral", kind=KIND_OPENAI,
            base_url="https://api.mistral.ai/v1", model="mistral-large-latest",
            api_key_name="MISTRAL_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="groq", label="Groq", kind=KIND_OPENAI,
            base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile",
            api_key_name="GROQ_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="together", label="Together AI", kind=KIND_OPENAI,
            base_url="https://api.together.xyz/v1", model="",
            api_key_name="TOGETHER_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="fireworks", label="Fireworks AI", kind=KIND_OPENAI,
            base_url="https://api.fireworks.ai/inference/v1", model="",
            api_key_name="FIREWORKS_API_KEY", enabled=False, builtin=True,
        ),
        ModelEndpoint(
            id="openrouter", label="OpenRouter", kind=KIND_OPENAI,
            base_url="https://openrouter.ai/api/v1", model="",
            api_key_name="OPENROUTER_API_KEY", enabled=False, builtin=True,
        ),
    ]


def builtin_endpoints() -> List[ModelEndpoint]:
    return _builtins()


# ── persistence ──────────────────────────────────────────────────────────────

def _config_path() -> Path:
    return config_path()


def _add_issue(issues: list[LoadIssue], issue: LoadIssue) -> None:
    if issue not in issues:
        issues.append(issue)


def _report(
    entries: Dict[str, ModelEndpoint],
    issues: list[LoadIssue],
) -> ModelConfigLoadReport:
    reasons = tuple(issues)
    return ModelConfigLoadReport(
        endpoints=tuple(entries.values()),
        complete=not reasons,
        reason_classes=reasons,
    )


def load_with_report(path: Path | None = None) -> ModelConfigLoadReport:
    """Load endpoints plus content-free completeness evidence.

    Merged rather than replaced so a new built-in (a lab that did not exist when
    the user last saved) appears without them having to reset anything, while
    their edits to an existing entry survive.

    Missing configuration is the valid built-in-only state.  Any present input
    that cannot be observed completely keeps every compatible endpoint that can
    be recovered, but marks the report incomplete.  This function performs no
    key lookup, readiness check, provider construction, or network operation.
    An injected path must be the exact native ``pathlib`` type; subclasses are
    refused so they cannot override the bounded read operation.
    """
    builtins = _builtins()
    entries: Dict[str, ModelEndpoint] = {
        endpoint.id: endpoint for endpoint in builtins
    }
    builtin_ids = frozenset(entries)
    issues: list[LoadIssue] = []
    selected = _config_path() if path is None else path
    if type(selected) is not _NATIVE_PATH_TYPE:
        _add_issue(issues, LoadIssue.UNREADABLE)
        return _report(entries, issues)

    try:
        with selected.open("rb") as source:
            encoded = source.read(MAX_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return _report(entries, issues)
    except Exception:  # noqa: BLE001
        _add_issue(issues, LoadIssue.UNREADABLE)
        return _report(entries, issues)

    if len(encoded) > MAX_CONFIG_BYTES:
        _add_issue(issues, LoadIssue.CORRUPT)
        return _report(entries, issues)
    try:
        raw = json.loads(encoded)
    except Exception:  # noqa: BLE001 -- this is the config containment boundary
        _add_issue(issues, LoadIssue.CORRUPT)
        return _report(entries, issues)

    if type(raw) is dict:
        if (
            any(type(key) is not str for key in raw)
            or set(raw) != {"version", "endpoints"}
            or type(raw.get("version")) is not int
            or raw.get("version") != CONFIG_SCHEMA_VERSION
        ):
            _add_issue(issues, LoadIssue.UNSUPPORTED_SCHEMA)
        candidate_rows = raw.get("endpoints")
        if type(candidate_rows) is not list:
            _add_issue(issues, LoadIssue.UNSUPPORTED_SCHEMA)
            rows: list[object] = []
        else:
            rows = candidate_rows
    elif type(raw) is list:
        # Legacy bare-list registries keep their historical load() behavior,
        # but cannot be called a complete observation of the versioned schema.
        _add_issue(issues, LoadIssue.UNSUPPORTED_ROOT)
        rows = raw
    else:
        _add_issue(issues, LoadIssue.UNSUPPORTED_ROOT)
        rows = []

    if len(rows) > MAX_ENDPOINT_ROWS:
        _add_issue(issues, LoadIssue.INVALID_ROW)
        rows = rows[:MAX_ENDPOINT_ROWS]

    fields = frozenset(ModelEndpoint.__dataclass_fields__)
    seen_ids: set[str] = set()
    for row in rows:
        if type(row) is not dict or any(type(key) is not str for key in row):
            _add_issue(issues, LoadIssue.INVALID_ROW)
            continue
        row_id = row.get("id")
        if not _bounded_text(row_id, nonempty=True):
            _add_issue(issues, LoadIssue.INVALID_ROW)
            continue
        if row_id in seen_ids:
            _add_issue(issues, LoadIssue.DUPLICATE_ENDPOINT_ID)
        seen_ids.add(row_id)
        if not set(row).issubset(fields):
            _add_issue(issues, LoadIssue.INVALID_ROW)
        if "builtin" in row and type(row["builtin"]) is not bool:
            _add_issue(issues, LoadIssue.INVALID_ROW)

        supplied = {
            key: value for key, value in row.items()
            if key in fields and key != "builtin"
        }
        merged = (
            {**asdict(entries[row_id]), **supplied}
            if row_id in entries
            else supplied
        )
        merged["builtin"] = row_id in builtin_ids
        try:
            entry = ModelEndpoint(**merged)
        except (TypeError, ValueError, UnicodeError):
            _add_issue(issues, LoadIssue.INVALID_ROW)
            continue
        entries[entry.id] = entry

    return _report(entries, issues)


def load() -> List[ModelEndpoint]:
    """Compatibility list view over :func:`load_with_report`.

    The assistant retains its historical fail-soft behavior: usable endpoint
    rows and built-ins remain available even when the diagnostic report is
    incomplete.  Persistence-sensitive observers must use the report directly.
    """
    return list(load_with_report().endpoints)


def save(endpoints: List[ModelEndpoint]) -> Path:
    """Persist the registry. Refuses to write a key VALUE.

    Written atomically: a half-written registry read at the next start would
    silently drop the user's endpoints.
    """
    path = _config_path()
    rows = []
    for e in endpoints:
        row = asdict(e)
        # Belt and braces. `ModelEndpoint` has no value field, but a future
        # edit adding one must not quietly start persisting secrets.
        for banned in ("api_key", "key", "secret", "token", "password"):
            row.pop(banned, None)
        rows.append(row)
    payload = json.dumps(
        {"version": CONFIG_SCHEMA_VERSION, "endpoints": rows},
        indent=2,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


def get(endpoint_id: str) -> Optional[ModelEndpoint]:
    for e in load():
        if e.id == endpoint_id:
            return e
    return None


def upsert(endpoint: ModelEndpoint) -> Path:
    entries = [e for e in load() if e.id != endpoint.id]
    entries.append(endpoint)
    return save(entries)


def remove(endpoint_id: str) -> Path:
    """Delete a user entry. A built-in is disabled instead of removed, because
    it would reappear from the catalogue on the next load and look like a bug."""
    entries = []
    for e in load():
        if e.id != endpoint_id:
            entries.append(e)
        elif e.builtin:
            e.enabled = False
            entries.append(e)
    return save(entries)


def ready_endpoints() -> List[ModelEndpoint]:
    """Configured, keyed, enabled — local first.

    Local before cloud for the same reason the original chain did it: it is
    free, private, and the book is on this machine. A cloud model is offered,
    never silently preferred.
    """
    ready = [e for e in load() if e.ready()]
    ready.sort(key=lambda e: (not e.local, e.label.lower()))
    return ready
