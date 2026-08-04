"""LLM providers for the Answer Engine — each returns an `llm_fn(prompt) -> str`
the engine can inject. Same seam as `panel/chat_panel.py`: local Ollama by default
(keyless), a Claude drop-in when `ANTHROPIC_API_KEY` is set. Kept tiny and
dependency-light; network failure degrades to an empty string so the engine simply
refuses rather than crashing."""
from __future__ import annotations

import json
import urllib.request
from typing import Callable


def normalize_ollama_chat_url(base_url: str) -> str:
    """Return one exact Ollama ``/api/chat`` endpoint.

    Runtime configuration historically mixed server roots, ``/api`` roots,
    and the full chat endpoint.  Appending blindly turns the configured default
    into ``/api/chat/api/chat`` and makes a healthy local server look offline.
    """
    base = str(base_url or "http://localhost:11434").strip().rstrip("/")
    if base.endswith("/api/chat"):
        return base
    if base.endswith("/api"):
        return base + "/chat"
    return base + "/api/chat"


def ollama_llm(base_url: str = "http://localhost:11434",
               model: str = "qwen3-coder:30b", *, temperature: float = 0.1,
               timeout: float = 300.0,
               max_tokens: int | None = None) -> Callable[[str], str]:
    """A local-Ollama author function. qwen3-coder authored 16/16 valid DSL
    programs first-try in the de-risk harness; temperature is low for determinism."""
    endpoint = normalize_ollama_chat_url(base_url)

    def _fn(prompt: str, schema=None) -> str:
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max(1, int(max_tokens))
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": options}
        if schema is not None:
            # Ollama compiles a JSON Schema into a llama.cpp GBNF grammar and
            # constrains sampling to it. This is the difference between asking
            # for a shape and being unable to emit anything else — the whole
            # basis of `assistant.constrain`.
            payload["format"] = schema
        try:
            req = urllib.request.Request(endpoint,
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message", {}) or {}).get("content", "") or data.get("response", "")
        except Exception:  # noqa: BLE001
            return ""      # engine will refuse honestly

    return _fn


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def anthropic_llm(model: str = "claude-sonnet-4-5-20250929", *,
                  max_tokens: int = 1200, temperature: float = 0.1,
                  timeout: float = 90.0) -> Callable[[str], str]:
    """The cloud drop-in, used when `ANTHROPIC_API_KEY` is present. Same
    contract as `ollama_llm`: a network failure returns "" so the caller refuses
    honestly rather than raising into a GUI thread.

    Note the identical seam is already in `atlas.fundamentals.guidance`; both
    exist because that one returns Optional[str] for its own provider chain.
    """
    def _fn(prompt: str, schema=None) -> str:
        # No grammar seam here: the cloud API constrains by tool-schema, not by
        # sampler. A constrained request therefore declines rather than
        # returning an UNCONSTRAINED answer that would be badged as guaranteed.
        if schema is not None:
            return ""
        try:
            from alelyon.runtime.atlas.data.keys import get_key
            key = get_key("ANTHROPIC_API_KEY")
        except Exception:  # noqa: BLE001
            key = None
        if not key:
            return ""
        body = json.dumps({"model": model, "max_tokens": max_tokens,
                           "temperature": temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        try:
            req = urllib.request.Request(
                _ANTHROPIC_URL, data=body,
                headers={"Content-Type": "application/json", "x-api-key": key,
                         "anthropic-version": _ANTHROPIC_VERSION})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parts = data.get("content") or []
            return "".join(p.get("text", "") for p in parts
                           if p.get("type") == "text").strip()
        except Exception:  # noqa: BLE001
            return ""

    return _fn


def have_anthropic_key() -> bool:
    try:
        from alelyon.runtime.atlas.data.keys import get_key
        return bool(get_key("ANTHROPIC_API_KEY"))
    except Exception:  # noqa: BLE001
        return False


# ── OpenAI-compatible: one wire format, many backends ────────────────────────
#
# `/v1/chat/completions` is the de-facto interface. vLLM, TGI, llama.cpp's
# server, LM Studio, Ollama's compatibility shim, OpenAI, Groq, Together,
# Fireworks, DeepSeek, Mistral, xAI and OpenRouter all speak it. One function
# therefore covers "my own fine-tuned weights served locally" and "a frontier
# lab" — the difference is a base URL and whether a key is attached, not a new
# code path per vendor.
#
# That matters beyond convenience: AGENTS.md §7 requires provider independence.
# A vendor-specific client per lab is how vendor assumptions leak into desks.

def normalize_openai_chat_url(base_url: str) -> str:
    """Return one exact `/v1/chat/completions` endpoint.

    Accepts a server root, a `/v1` root, or the full endpoint, because every
    one of those is what some backend's own documentation prints. Appending
    blindly produced `/v1/chat/completions/v1/chat/completions` and made a
    healthy server look unreachable — the same defect
    `normalize_ollama_chat_url` exists to prevent.
    """
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def openai_compatible_llm(
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    api_key_name: str = "",
    temperature: float = 0.1,
    timeout: float = 300.0,
    max_tokens: int | None = None,
    organization: str = "",
) -> Callable[[str], str]:
    """An `llm_fn` for any OpenAI-compatible endpoint.

    `api_key_name` is resolved through `keys.get_key()` at CALL time rather than
    captured at construction, so a key added while the application is running
    takes effect without a restart — and so the key is never held in a closure
    any longer than the request that uses it.

    Contract matches `ollama_llm`: any failure returns "" so the caller refuses
    honestly instead of raising into a GUI thread.

    No grammar seam. The OpenAI wire format constrains via `response_format` /
    tool schemas, not a sampler grammar, and `assistant.constrain` depends on
    the sampler being unable to emit anything off-shape. Claiming grammar
    support here would badge an unconstrained answer as guaranteed, so a
    constrained request declines — exactly as `anthropic_llm` does.
    """
    endpoint = normalize_openai_chat_url(base_url)

    def _fn(prompt: str, schema=None) -> str:
        if schema is not None:
            return ""
        if not endpoint:
            return ""
        key = api_key
        if not key and api_key_name:
            try:
                from alelyon.runtime.atlas.data.keys import get_key
                key = get_key(api_key_name) or ""
            except Exception:  # noqa: BLE001
                key = ""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max(1, int(max_tokens))
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if organization:
            headers["OpenAI-Organization"] = organization
        try:
            req = urllib.request.Request(
                endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            content = message.get("content")
            # Some backends return content as a list of typed parts.
            if isinstance(content, list):
                return "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") in (None, "text")
                ).strip()
            return (content or "").strip()
        except Exception:  # noqa: BLE001
            return ""      # engine will refuse honestly

    return _fn
