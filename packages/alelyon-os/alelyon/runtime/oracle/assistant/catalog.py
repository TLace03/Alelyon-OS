"""What models a configured server actually has, asked at the moment it matters.

The endpoint registry's model names are declarations, and the module that owns
them says so plainly: vendors rename and retire models, so every built-in name
is a starting point the user is expected to edit. That was the honest position
while the only alternative was a hard-coded list that would itself go stale.

Both server families in the registry can simply be asked:

* **Ollama** publishes its installed models at `GET /api/tags`.
* **The OpenAI-compatible world** — vLLM, LM Studio, llama.cpp's server, and
  the hosted labs — publishes what it serves at `GET /v1/models`.

A stale name therefore becomes self-healing: the surface offering a model list
offers what the server said it has, not what a file remembered.

What a listing is, and is not
-----------------------------
A row here means the server CLAIMS to serve that name. It is availability, not
correctness: nothing about a listed name proves the model loads, answers, or
answers well — that is what a probe measures. And an empty listing with a
refusal is a different fact from an empty listing without one: "the server is
down" and "the server has no models" must never collapse into each other,
which is why this module returns a reading with a named refusal rather than a
bare list.

Never called at import, never called implicitly. A registry load must stay a
file read; only a user-visible surface (a picker opening, a probe button)
should spend a network round trip, even a loopback one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TIMEOUT = 5.0
MAX_LISTING_BYTES = 4_194_304
MAX_LISTED_MODELS = 2_000

#: A hex digest long enough that nobody names a model after one by accident.
#: Ollama's own blob rows carry sha256, so 40 admits sha1-shaped addresses too
#: without reaching down to anything a person would type.
_MIN_DIGEST_CHARS = 40
_DIGEST_RE = re.compile(
    r"^(?:sha\d{0,3}[:\-])?[0-9a-f]{%d,}$" % _MIN_DIGEST_CHARS)


def is_content_digest(name: str) -> bool:
    """Whether a listed row names a content address rather than a model.

    Measured on this workstation 2026-08-22: ``GET /api/tags`` returned three
    rows named ``blobs:sha256-<64 hex>`` with ``size`` 0, alongside the five
    real models. The server publishes them, so every parser of that endpoint
    forwards them faithfully, and the Models view listed them as installed
    models a user could select. A digest is not a name: nothing can be pulled,
    shown, or served by it.

    The test is on the NAME rather than on ``size == 0``, deliberately. A
    zero-byte row is a plausible transient for a real model mid-pull, and
    dropping it would hide a model the user is waiting for; a name that is a
    bare content address cannot be a model under any state of the server.
    """
    text = str(name or "").strip().lower()
    if not text:
        return False
    repo, _, tag = text.partition(":")
    if repo == "blobs":
        return True
    return bool(_DIGEST_RE.match(tag or repo))


@dataclass(frozen=True)
class ListedModel:
    """One name a server claims to serve."""

    name: str
    #: Bytes on disk, when the server reported a size (Ollama does). None is
    #: absent, not zero.
    size_bytes: Optional[int] = None
    #: The server's own modification stamp, verbatim, when it gave one.
    modified: str = ""


@dataclass(frozen=True)
class CatalogReading:
    """A listing plus the evidence of how complete it is.

    `refusal` is empty only when the server answered and the answer parsed.
    A reading with a refusal may still carry rows — a truncated or partially
    parseable answer keeps what could be recovered — but a caller must show
    the refusal rather than present the fragment as the whole catalogue.
    """

    models: Tuple[ListedModel, ...]
    refusal: str = ""
    #: Rows the server published that are not models — content digests today.
    #: Carried rather than discarded silently: a listing that drops rows
    #: without saying so cannot be reconciled against `ollama list`, and the
    #: reader is left to wonder whether the tool or the server lost them.
    dropped: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.refusal


def _read_json(url: str, *, headers: Optional[dict] = None,
               timeout: float = DEFAULT_TIMEOUT):
    """(parsed, refusal). Bounded read; any failure is a named refusal."""
    import urllib.request
    from alelyon.runtime.oracle.answer.providers import oracle_urlopen

    request = urllib.request.Request(url, headers=headers or {})
    try:
        with oracle_urlopen(request, timeout=timeout) as resp:
            encoded = resp.read(MAX_LISTING_BYTES + 1)
    except Exception as exc:  # noqa: BLE001
        return None, f"the server could not be reached ({type(exc).__name__})"
    if len(encoded) > MAX_LISTING_BYTES:
        return None, "the server's answer exceeded the size bound"
    try:
        return json.loads(encoded.decode("utf-8")), ""
    except Exception:  # noqa: BLE001
        return None, "the server's answer was not JSON"


def _int_or_none(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def ollama_models_url(base_url: str) -> str:
    """The `/api/tags` endpoint for a configured Ollama base URL.

    Derived from the same normalisation the chat transport uses, so the two
    can never disagree about which server "the configured one" is.
    """
    from alelyon.runtime.oracle.answer.providers import normalize_ollama_chat_url

    chat_endpoint = normalize_ollama_chat_url(base_url)
    # …/api/chat -> …/api/tags, by construction of the normaliser.
    return chat_endpoint[: -len("/chat")] + "/tags"


def openai_models_url(base_url: str) -> str:
    """The `/v1/models` endpoint beside a configured chat endpoint."""
    from alelyon.runtime.oracle.answer.providers import normalize_openai_chat_url

    chat_endpoint = normalize_openai_chat_url(base_url)
    if not chat_endpoint:
        return ""
    # …/chat/completions -> …/models, again by construction.
    return chat_endpoint[: -len("/chat/completions")] + "/models"


def ollama_installed_models(base_url: str = "", *,
                            timeout: float = DEFAULT_TIMEOUT) -> CatalogReading:
    """What this Ollama server has pulled, or a named refusal."""
    import os

    base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    raw, refusal = _read_json(ollama_models_url(base), timeout=timeout)
    if refusal:
        return CatalogReading(models=(), refusal=refusal)
    rows = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return CatalogReading(models=(),
                              refusal="the server's answer had no model list")
    out: list[ListedModel] = []
    dropped: list[str] = []
    for row in rows[:MAX_LISTED_MODELS]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("model") or "").strip()
        if not name:
            continue
        if is_content_digest(name):
            dropped.append(name)
            continue
        out.append(ListedModel(name=name,
                               size_bytes=_int_or_none(row.get("size")),
                               modified=str(row.get("modified_at") or "")))
    return CatalogReading(models=tuple(out), dropped=tuple(dropped))


def openai_listed_models(base_url: str, *, api_key_name: str = "",
                         timeout: float = DEFAULT_TIMEOUT) -> CatalogReading:
    """What an OpenAI-compatible server claims to serve, or a named refusal.

    The key is resolved through `keys.get_key()` at call time and travels only
    in the request header — the same handling every transport in this package
    gives it.
    """
    url = openai_models_url(base_url)
    if not url:
        return CatalogReading(models=(), refusal="no endpoint is configured")
    headers = {}
    if api_key_name:
        try:
            from alelyon.runtime.atlas.data.keys import get_key
            key = get_key(api_key_name) or ""
        except Exception:  # noqa: BLE001
            key = ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
    raw, refusal = _read_json(url, headers=headers, timeout=timeout)
    if refusal:
        return CatalogReading(models=(), refusal=refusal)
    rows = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        # Some servers answer with a bare list; accept it rather than refusing
        # a working server over an envelope.
        rows = raw if isinstance(raw, list) else None
    if not isinstance(rows, list):
        return CatalogReading(models=(),
                              refusal="the server's answer had no model list")
    out: list[ListedModel] = []
    for row in rows[:MAX_LISTED_MODELS]:
        if isinstance(row, dict):
            name = str(row.get("id") or row.get("name") or "").strip()
        else:
            name = str(row or "").strip()
        if name:
            out.append(ListedModel(name=name))
    return CatalogReading(models=tuple(out))


def hostname_of(base_url: str) -> str:
    """The host a listing call would reach, for a surface that shows it."""
    try:
        return (urlsplit(str(base_url or "").strip()).hostname or "").strip("[]")
    except ValueError:
        return ""


def redacted_url(base_url: str) -> str:
    """The URL with any userinfo stripped, safe for a log line or a label."""
    try:
        parts = urlsplit(str(base_url or "").strip())
    except ValueError:
        return ""
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@dataclass(frozen=True)
class ProbeResult:
    """Whether an endpoint's configured model actually answered a minimal turn.

    A listing asks what a server HAS; a probe asks whether it ANSWERS — the
    distinction the picker cannot make from config alone. `detail` is a
    human-readable reason either way. `latency_s` is None when no round trip
    completed, so a failure never reads as an instant success.
    """

    ok: bool
    detail: str
    latency_s: Optional[float] = None


def probe_endpoint(endpoint, *, chat=None, clock=None) -> ProbeResult:
    """Send one minimal turn through the endpoint's own transport and MEASURE it.

    Not a guess from the config: the reply is the evidence. `chat` and `clock`
    are injected so a test drives this with no network and a fixed latency; by
    default the transport is built from the endpoint exactly as the chat
    workspace builds it, so a probe exercises the path a real answer will take.
    """
    import time as _time

    from alelyon.runtime.oracle.answer.chat import ChatMessage

    clock = clock or _time.monotonic
    if chat is None:
        from alelyon.runtime.oracle.assistant.providers import endpoint_provider
        chat = endpoint_provider(endpoint).chat
    started = clock()
    try:
        reply = chat([ChatMessage("user", "ping")], None, None)
    except Exception as exc:  # noqa: BLE001 — a dead endpoint must not raise into the GUI
        return ProbeResult(False, f"no answer ({type(exc).__name__})")
    latency = float(clock() - started)
    if getattr(reply, "error", ""):
        return ProbeResult(False, reply.error, latency)
    if not (getattr(reply, "text", "") or "").strip():
        return ProbeResult(False, "the model returned nothing", latency)
    return ProbeResult(True, f"answered in {latency:.1f}s", latency)
