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
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from alelyon.runtime.common.paths import GLOBALS_DIR

#: Where the registry persists. State, not source — it travels with the user.
CONFIG_PATH: Path = GLOBALS_DIR / "model_endpoints.json"

KIND_OLLAMA = "ollama"
KIND_OPENAI = "openai-compatible"
KIND_ANTHROPIC = "anthropic"
KINDS = (KIND_OLLAMA, KIND_OPENAI, KIND_ANTHROPIC)

#: An api_key_name must look like an environment variable, because that is what
#: it becomes. Refusing anything else keeps a crafted name out of the env file.
_KEY_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


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
        if self.kind not in KINDS:
            raise ValueError(f"endpoint {self.id!r} has unknown kind {self.kind!r}")
        if self.api_key_name and not _KEY_NAME_RE.match(self.api_key_name):
            raise ValueError(
                f"endpoint {self.id!r} has an api_key_name that is not a valid "
                f"environment-variable name: {self.api_key_name!r}")

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
    override = (os.environ.get("ALELYON_MODEL_CONFIG") or "").strip()
    return Path(override) if override else CONFIG_PATH


def load() -> List[ModelEndpoint]:
    """Every endpoint: built-ins merged with the user's saved edits.

    Merged rather than replaced so a new built-in (a lab that did not exist when
    the user last saved) appears without them having to reset anything, while
    their edits to an existing entry survive.

    A corrupt or unreadable file yields the built-ins. Losing customisation is
    bad; refusing to start the assistant because a JSON file has a stray comma
    is worse.
    """
    entries: Dict[str, ModelEndpoint] = {e.id: e for e in _builtins()}
    try:
        raw = json.loads(_config_path().read_text(encoding="utf-8"))
        rows = raw.get("endpoints") if isinstance(raw, dict) else raw
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            merged = {**asdict(entries[row["id"]]), **row} if row["id"] in entries else row
            merged.pop("builtin", None)
            try:
                entry = ModelEndpoint(**{
                    k: v for k, v in merged.items()
                    if k in ModelEndpoint.__dataclass_fields__ and k != "builtin"
                })
            except (TypeError, ValueError):
                continue          # one bad row never discards the rest
            entry.builtin = row["id"] in {e.id for e in _builtins()}
            entries[entry.id] = entry
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return list(entries.values())


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
    payload = json.dumps({"version": 1, "endpoints": rows}, indent=2) + "\n"
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
