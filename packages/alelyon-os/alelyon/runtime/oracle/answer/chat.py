"""Conversation-native transport: the same three wire formats, real messages.

`providers.py` and `streaming.py` give every backend one shape — a flat prompt
string in, text out. That is the right contract for the Answer Engine's
one-shot structured requests (routing, constrained decoding), and it is the
wrong one for a conversation: every backend on this machine natively accepts a
message array with roles, and flattening a dialogue into one string discards
the structure the model was trained to read. A system prompt rendered as the
first paragraph of a user message is not a system prompt; the model treats it
as something the user said.

This module adds the second shape without disturbing the first:

    chat(messages) -> ChatReply
    chat_stream(messages, sink, cancel=None) -> ChatReply

`messages` is a sequence of typed `ChatMessage` records with a closed role
vocabulary. Each wire format receives them in its own native representation:

* **OpenAI-compatible** — the array as-is, `system` as a message role.
* **Anthropic** — system content lifted into the top-level `system` parameter,
  which is that API's native representation; user/assistant turns in the array.
* **Ollama** — the native `/api/chat` messages array, `system` role included.

Token accounting rides along where the wire reports it. `TokenUsage` fields
are `None` when the backend did not report a count — absent is absent, never
zero, because a zero would read as "measured: nothing" and that is a different
claim (AGENTS.md §14: an absent term is UNMEASURED, never silently zero).

`flatten_messages` is the ONE place a conversation is rendered to a single
prompt, for backends driven through the legacy seam. Two independent
flattenings would disagree about what the model saw, and which rendition ran
is exactly the thing a saved transcript needs to be able to say. A reply
produced through that fallback carries `native=False`.

Contract matches the sibling modules: nothing here raises. A network failure,
a missing key, a malformed frame, a cancellation and a size runaway all come
back as a `ChatReply` with a stated reason, because this seam exists to be
called from a GUI thread.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from alelyon.runtime.oracle.answer.providers import (
    _ANTHROPIC_URL, _ANTHROPIC_VERSION, normalize_ollama_chat_url,
    normalize_openai_chat_url, oracle_urlopen,
)
from alelyon.runtime.oracle.answer.streaming import (
    MAX_STREAM_CHARS, CancelCheck, StreamSink, sse_events,
)

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
#: The closed role vocabulary. Closed because every wire format below has to
#: place each role somewhere specific, and an unknown role would have to be
#: guessed into one of these slots — silently, per format, differently.
ROLES: Tuple[str, ...] = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT)

#: Ceiling on one message's content. Generous — whole files get pasted into
#: conversations — but present, because an unbounded field here becomes an
#: unbounded request body built in a GUI process.
MAX_MESSAGE_CHARS = 400_000


@dataclass(frozen=True)
class ChatMessage:
    """One turn, as the wire will see it."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown chat role: {self.role!r}")
        if type(self.content) is not str:
            raise TypeError("message content must be a string")
        if len(self.content) > MAX_MESSAGE_CHARS:
            raise ValueError("message content exceeds its size bound")


@dataclass(frozen=True)
class TokenUsage:
    """What the backend said this exchange cost, when it said anything.

    `None` means the wire carried no count — UNMEASURED, which is a different
    fact from zero. Streaming responses on some backends report nothing, and a
    caller summing usage across turns must skip absent readings rather than
    add a zero that looks like a measurement.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def measured(self) -> bool:
        return self.prompt_tokens is not None or self.completion_tokens is not None


@dataclass(frozen=True)
class ChatReply:
    """What came back, and on what authority.

    The shape deliberately parallels `streaming.StreamResult`: `complete` is
    True only when the format's own end marker was seen, `error` is a stated
    human-readable reason, and `cancelled` records the user's decision as
    distinct from a failure.
    """

    text: str
    complete: bool
    error: str = ""
    cancelled: bool = False
    #: The backend's own stop reason, verbatim, when it gave one ("stop",
    #: "length", "end_turn", …). Vendor vocabulary, deliberately not mapped to
    #: a shared enum — the words are one vendor's and a mapping would be a
    #: guess repeated per vendor.
    finish_reason: str = ""
    #: Token accounting, when the wire reported it. Never invented.
    usage: Optional[TokenUsage] = None
    #: Whether the backend saw the message array natively. False means the
    #: conversation was flattened to one prompt through the legacy seam — the
    #: answer is real, but the model read a rendition, and a transcript must
    #: be able to say which.
    native: bool = True

    @property
    def ok(self) -> bool:
        return self.complete and not self.error and not self.cancelled


#: One conversation in, one finished reply out.
ChatFn = Callable[[Sequence[ChatMessage]], ChatReply]
#: The incremental counterpart: fragments to the sink as they arrive.
ChatStreamFn = Callable[[Sequence[ChatMessage], StreamSink, CancelCheck], ChatReply]


def flatten_messages(messages: Sequence[ChatMessage]) -> str:
    """Render a conversation to a single prompt, deterministically.

    THE one flattening. The legacy `llm(prompt)` seam and any transcript that
    needs to show "what the model was actually sent" both call this, so there
    is exactly one answer to that question.

    System content leads, unlabelled — it is instruction, not dialogue. Turns
    follow with plain-language labels. The rendition ends after the last
    message with no trailing cue: the final message is the user's, and the
    model continues from there exactly as it would with a bare prompt.
    """
    system: list[str] = []
    turns: list[str] = []
    for message in messages:
        if message.role == ROLE_SYSTEM:
            if message.content.strip():
                system.append(message.content.strip())
        else:
            label = "User" if message.role == ROLE_USER else "Assistant"
            turns.append(f"{label}: {message.content}")
    parts = []
    if system:
        parts.append("\n\n".join(system))
    if turns:
        parts.append("\n\n".join(turns))
    return "\n\n".join(parts)


def _clean(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Validated copies, in order. Raises on a malformed sequence — that is a
    caller bug, not a network condition, and must not be reported as one."""
    out: list[ChatMessage] = []
    for message in messages:
        if type(message) is not ChatMessage:
            message = ChatMessage(role=getattr(message, "role", ""),
                                  content=str(getattr(message, "content", "")))
        out.append(message)
    return out


def _split_system(messages: Sequence[ChatMessage]) -> Tuple[str, list[dict]]:
    """(system_text, dialogue_rows) for the Anthropic representation."""
    system: list[str] = []
    rows: list[dict] = []
    for message in messages:
        if message.role == ROLE_SYSTEM:
            if message.content.strip():
                system.append(message.content.strip())
        else:
            rows.append({"role": message.role, "content": message.content})
    return "\n\n".join(system), rows


def _rows(messages: Sequence[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _key_for(api_key: str, api_key_name: str) -> str:
    key = api_key
    if not key and api_key_name:
        try:
            from alelyon.runtime.atlas.data.keys import get_key
            key = get_key(api_key_name) or ""
        except Exception:  # noqa: BLE001
            key = ""
    return key


def _open(url: str, body: dict, headers: dict, timeout: float):
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers)
    return oracle_urlopen(request, timeout=timeout)


def _unreachable(exc: Exception) -> ChatReply:
    return ChatReply("", False, error=f"the model could not be reached "
                                      f"({type(exc).__name__})")


def _int_or_none(value) -> Optional[int]:
    """A count the wire actually stated, or None. Never coerces absence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


# ── usage extraction, one small parser per wire format ───────────────────────
def openai_usage(obj: dict) -> Optional[TokenUsage]:
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(usage, dict):
        return None
    reading = TokenUsage(prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                         completion_tokens=_int_or_none(usage.get("completion_tokens")))
    return reading if reading.measured else None


def anthropic_usage(obj: dict) -> Optional[TokenUsage]:
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(usage, dict):
        return None
    reading = TokenUsage(prompt_tokens=_int_or_none(usage.get("input_tokens")),
                         completion_tokens=_int_or_none(usage.get("output_tokens")))
    return reading if reading.measured else None


def ollama_usage(obj: dict) -> Optional[TokenUsage]:
    if not isinstance(obj, dict):
        return None
    reading = TokenUsage(prompt_tokens=_int_or_none(obj.get("prompt_eval_count")),
                         completion_tokens=_int_or_none(obj.get("eval_count")))
    return reading if reading.measured else None


def _merge_usage(base: Optional[TokenUsage],
                 patch: Optional[TokenUsage]) -> Optional[TokenUsage]:
    """Later readings fill gaps; a stated count is never overwritten by None.

    Anthropic reports input tokens at `message_start` and output tokens at
    `message_delta`, so a streamed exchange assembles its usage from two
    events. Each side keeps the last value the wire actually stated.
    """
    if patch is None:
        return base
    if base is None:
        return patch
    return TokenUsage(
        prompt_tokens=patch.prompt_tokens if patch.prompt_tokens is not None
        else base.prompt_tokens,
        completion_tokens=patch.completion_tokens
        if patch.completion_tokens is not None else base.completion_tokens,
    )


# ── OpenAI-compatible ────────────────────────────────────────────────────────
def openai_chat(base_url: str, model: str, *, api_key: str = "",
                api_key_name: str = "", temperature: float = 0.1,
                timeout: float = 300.0, max_tokens: Optional[int] = None,
                organization: str = "") -> ChatFn:
    """A conversation callable for any OpenAI-compatible endpoint.

    The key is resolved at call time, exactly as `providers.openai_compatible_llm`
    resolves it and for the same reasons: a key entered while the application
    runs takes effect without a restart, and it never sits in a closure longer
    than the request that uses it.
    """
    endpoint = normalize_openai_chat_url(base_url)

    def _chat(messages: Sequence[ChatMessage]) -> ChatReply:
        if not endpoint:
            return ChatReply("", False, error="no endpoint is configured")
        rows = _rows(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        body = {"model": model, "messages": rows,
                "temperature": temperature, "stream": False}
        if max_tokens is not None:
            body["max_tokens"] = max(1, int(max_tokens))
        headers = {"Content-Type": "application/json"}
        key = _key_for(api_key, api_key_name)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if organization:
            headers["OpenAI-Organization"] = organization
        try:
            with _open(endpoint, body, headers, timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return ChatReply("", False, error="the model returned no choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") in (None, "text"))
        text = (content or "").strip()
        if not text:
            return ChatReply("", False, error="the model returned nothing",
                             usage=openai_usage(data))
        return ChatReply(text, True,
                         finish_reason=str(first.get("finish_reason") or ""),
                         usage=openai_usage(data))

    return _chat


def openai_chat_stream(base_url: str, model: str, *, api_key: str = "",
                       api_key_name: str = "", temperature: float = 0.1,
                       timeout: float = 300.0,
                       max_tokens: Optional[int] = None,
                       organization: str = "") -> ChatStreamFn:
    """Streaming counterpart. Usage is parsed from any chunk that volunteers a
    `usage` object; no `stream_options` flag is sent, because several
    OpenAI-compatible servers reject request fields they do not know, and a
    conversation that breaks on an accounting flag has its priorities inverted.
    Absent usage stays absent."""
    endpoint = normalize_openai_chat_url(base_url)

    def _stream(messages: Sequence[ChatMessage], sink: StreamSink,
                cancel: CancelCheck = None) -> ChatReply:
        if not endpoint:
            return ChatReply("", False, error="no endpoint is configured")
        rows = _rows(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        body = {"model": model, "messages": rows,
                "temperature": temperature, "stream": True}
        if max_tokens is not None:
            body["max_tokens"] = max(1, int(max_tokens))
        headers = {"Content-Type": "application/json"}
        key = _key_for(api_key, api_key_name)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if organization:
            headers["OpenAI-Organization"] = organization

        parts: list[str] = []
        total = 0
        usage: Optional[TokenUsage] = None
        finish = ""
        try:
            with _open(endpoint, body, headers, timeout) as response:
                def _lines():
                    for raw in response:
                        if cancel is not None and cancel():
                            return
                        yield (raw.decode("utf-8", "replace")
                               if isinstance(raw, bytes) else str(raw))

                for _event, data in sse_events(_lines()):
                    payload = (data or "").strip()
                    if not payload:
                        continue
                    if payload == "[DONE]":
                        return ChatReply("".join(parts), True,
                                         finish_reason=finish, usage=usage)
                    try:
                        obj = json.loads(payload)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(obj, dict):
                        continue
                    usage = _merge_usage(usage, openai_usage(obj))
                    choices = obj.get("choices")
                    first = (choices[0] if isinstance(choices, list) and choices
                             and isinstance(choices[0], dict) else {})
                    delta = (first.get("delta")
                             if isinstance(first.get("delta"), dict) else {})
                    content = delta.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") in (None, "text"))
                    fragment = str(content or "")
                    if fragment:
                        parts.append(fragment)
                        total += len(fragment)
                        try:
                            sink(fragment)
                        except Exception:  # noqa: BLE001
                            pass
                        if total >= MAX_STREAM_CHARS:
                            return ChatReply(
                                "".join(parts), False, usage=usage,
                                error=f"the answer passed {MAX_STREAM_CHARS:,} "
                                      f"characters and was cut off")
                    if first.get("finish_reason"):
                        finish = str(first["finish_reason"])
                        # `finish_reason` closes the choice; several servers
                        # send only it and never `[DONE]`.
                        return ChatReply("".join(parts), True,
                                         finish_reason=finish, usage=usage)
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        if cancel is not None and cancel():
            return ChatReply("".join(parts), False, cancelled=True, usage=usage)
        return ChatReply("".join(parts), False, usage=usage,
                         error="the model's connection closed before it finished")

    return _stream


# ── Anthropic ────────────────────────────────────────────────────────────────
def anthropic_chat(model: str, *, max_tokens: int = 1200,
                   temperature: float = 0.1, timeout: float = 90.0,
                   api_key_name: str = "ANTHROPIC_API_KEY") -> ChatFn:
    """The Anthropic messages API, with system content in its native slot."""

    def _chat(messages: Sequence[ChatMessage]) -> ChatReply:
        key = _key_for("", api_key_name)
        if not key:
            return ChatReply("", False, error=f"no {api_key_name} is set")
        system, rows = _split_system(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        body = {"model": model, "max_tokens": max_tokens,
                "temperature": temperature, "messages": rows}
        if system:
            body["system"] = system
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": _ANTHROPIC_VERSION}
        try:
            with _open(_ANTHROPIC_URL, body, headers, timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        parts = data.get("content") if isinstance(data, dict) else None
        text = "".join(p.get("text", "") for p in (parts or [])
                       if isinstance(p, dict) and p.get("type") == "text").strip()
        if not text:
            return ChatReply("", False, error="the model returned nothing",
                             usage=anthropic_usage(data if isinstance(data, dict) else {}))
        return ChatReply(text, True,
                         finish_reason=str(data.get("stop_reason") or ""),
                         usage=anthropic_usage(data))

    return _chat


def anthropic_chat_stream(model: str, *, max_tokens: int = 1200,
                          temperature: float = 0.1, timeout: float = 90.0,
                          api_key_name: str = "ANTHROPIC_API_KEY") -> ChatStreamFn:
    """Streaming messages API. Usage assembles from two events: input tokens
    at `message_start`, output tokens at `message_delta`."""

    def _stream(messages: Sequence[ChatMessage], sink: StreamSink,
                cancel: CancelCheck = None) -> ChatReply:
        key = _key_for("", api_key_name)
        if not key:
            return ChatReply("", False, error=f"no {api_key_name} is set")
        system, rows = _split_system(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        body = {"model": model, "max_tokens": max_tokens, "stream": True,
                "temperature": temperature, "messages": rows}
        if system:
            body["system"] = system
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": _ANTHROPIC_VERSION}

        parts: list[str] = []
        total = 0
        usage: Optional[TokenUsage] = None
        finish = ""
        try:
            with _open(_ANTHROPIC_URL, body, headers, timeout) as response:
                def _lines():
                    for raw in response:
                        if cancel is not None and cancel():
                            return
                        yield (raw.decode("utf-8", "replace")
                               if isinstance(raw, bytes) else str(raw))

                for event, data in sse_events(_lines()):
                    payload = (data or "").strip()
                    obj = None
                    if payload:
                        try:
                            obj = json.loads(payload)
                        except Exception:  # noqa: BLE001
                            obj = None
                    name = (event or "").strip()
                    if not name and isinstance(obj, dict):
                        name = str(obj.get("type") or "")
                    if name == "message_start" and isinstance(obj, dict):
                        inner = obj.get("message")
                        if isinstance(inner, dict):
                            usage = _merge_usage(usage, anthropic_usage(inner))
                        continue
                    if name == "message_delta" and isinstance(obj, dict):
                        usage = _merge_usage(usage, anthropic_usage(obj))
                        delta = obj.get("delta")
                        if isinstance(delta, dict) and delta.get("stop_reason"):
                            finish = str(delta["stop_reason"])
                        continue
                    if name == "message_stop":
                        return ChatReply("".join(parts), True,
                                         finish_reason=finish, usage=usage)
                    if name == "error":
                        return ChatReply("".join(parts), False, usage=usage,
                                         error="the model reported an error "
                                               "mid-answer")
                    if name != "content_block_delta" or not isinstance(obj, dict):
                        continue
                    delta = obj.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    if delta.get("type") not in (None, "text_delta"):
                        continue
                    fragment = str(delta.get("text") or "")
                    if fragment:
                        parts.append(fragment)
                        total += len(fragment)
                        try:
                            sink(fragment)
                        except Exception:  # noqa: BLE001
                            pass
                        if total >= MAX_STREAM_CHARS:
                            return ChatReply(
                                "".join(parts), False, usage=usage,
                                error=f"the answer passed {MAX_STREAM_CHARS:,} "
                                      f"characters and was cut off")
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        if cancel is not None and cancel():
            return ChatReply("".join(parts), False, cancelled=True, usage=usage)
        return ChatReply("".join(parts), False, usage=usage,
                         error="the model's connection closed before it finished")

    return _stream


# ── Ollama ───────────────────────────────────────────────────────────────────
def ollama_chat(base_url: str, model: str, *, temperature: float = 0.1,
                timeout: float = 300.0,
                max_tokens: Optional[int] = None) -> ChatFn:
    """The native `/api/chat`, message array included. Ollama accepts the
    `system` role inside the array, so no lifting is needed."""
    endpoint = normalize_ollama_chat_url(base_url)

    def _chat(messages: Sequence[ChatMessage]) -> ChatReply:
        rows = _rows(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max(1, int(max_tokens))
        body = {"model": model, "messages": rows, "stream": False,
                "options": options}
        try:
            with _open(endpoint, body, {"Content-Type": "application/json"},
                       timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        if not isinstance(data, dict):
            return ChatReply("", False, error="the model returned nothing")
        message = data.get("message")
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        text = (text or str(data.get("response") or "")).strip()
        if not text:
            return ChatReply("", False, error="the model returned nothing",
                             usage=ollama_usage(data))
        return ChatReply(text, True,
                         finish_reason=str(data.get("done_reason") or ""),
                         usage=ollama_usage(data))

    return _chat


def ollama_chat_stream(base_url: str, model: str, *, temperature: float = 0.1,
                       timeout: float = 300.0,
                       max_tokens: Optional[int] = None) -> ChatStreamFn:
    """Streaming `/api/chat`. The final `done: true` object carries the counts."""
    endpoint = normalize_ollama_chat_url(base_url)

    def _stream(messages: Sequence[ChatMessage], sink: StreamSink,
                cancel: CancelCheck = None) -> ChatReply:
        rows = _rows(_clean(messages))
        if not rows:
            return ChatReply("", False, error="the conversation is empty")
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max(1, int(max_tokens))
        body = {"model": model, "messages": rows, "stream": True,
                "options": options}
        parts: list[str] = []
        total = 0
        try:
            with _open(endpoint, body, {"Content-Type": "application/json"},
                       timeout) as response:
                for raw in response:
                    if cancel is not None and cancel():
                        return ChatReply("".join(parts), False, cancelled=True)
                    line = (raw.decode("utf-8", "replace")
                            if isinstance(raw, bytes) else str(raw)).strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(obj, dict):
                        continue
                    message = obj.get("message")
                    fragment = ""
                    if isinstance(message, dict):
                        fragment = str(message.get("content") or "")
                    if not fragment:
                        fragment = str(obj.get("response") or "")
                    if fragment:
                        parts.append(fragment)
                        total += len(fragment)
                        try:
                            sink(fragment)
                        except Exception:  # noqa: BLE001
                            pass
                        if total >= MAX_STREAM_CHARS:
                            return ChatReply(
                                "".join(parts), False,
                                error=f"the answer passed {MAX_STREAM_CHARS:,} "
                                      f"characters and was cut off")
                    if obj.get("done"):
                        return ChatReply(
                            "".join(parts), True,
                            finish_reason=str(obj.get("done_reason") or ""),
                            usage=ollama_usage(obj))
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        if cancel is not None and cancel():
            return ChatReply("".join(parts), False, cancelled=True)
        return ChatReply("".join(parts), False,
                         error="the model's connection closed before it finished")

    return _stream
