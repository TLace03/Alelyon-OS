"""Incremental generation: the same providers, delivered a token at a time.

`providers.py` gives every backend one shape — `llm(prompt) -> str`, empty on
failure — and that shape is deliberately blind to time. It is the right contract
for the Answer Engine, where the caller wants a finished program or nothing. It
is the wrong one for a conversation, where a minute of a blank panel is
indistinguishable from a hang and the user's only recourse is to kill the app.

This module adds the second shape without disturbing the first:

    stream(prompt, sink, cancel=None) -> StreamResult

The sink is called with each fragment as it arrives; the return value carries
the whole text plus **whether the backend actually finished**. That second field
is the reason this is a record and not a string. Three wire formats all fail the
same way — the socket closes — and a closed socket after 200 tokens looks
exactly like a model that chose to stop after 200 tokens. Only the format's own
end-of-stream marker distinguishes them, so it is read and reported rather than
assumed. A caller that shows a truncated answer as a complete one is publishing
the model's first half as its conclusion.

Three formats, three decoders, one loop:

* **Ollama** — newline-delimited JSON, one object per fragment, `done: true` last.
* **OpenAI-compatible** — Server-Sent Events, `choices[0].delta.content`,
  terminated by the literal `data: [DONE]`.
* **Anthropic** — Server-Sent Events, `content_block_delta` fragments,
  terminated by `message_stop`.

The decoders are pure functions over already-decoded lines, so the parsing —
which is where wire formats actually break — is testable without a socket.

Nothing here raises. A network failure, a malformed frame, a cancellation and a
size runaway all come back as a `StreamResult` with `complete=False` and a
stated reason, because a provider that raises into a GUI thread is a crash and
this seam exists to be called from one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Tuple

#: Hard ceiling on a single streamed answer. A model stuck in a repetition loop
#: emits fragments forever, and the accumulator on the other end is a GUI
#: process's memory. Hitting this is a truncation and is reported as one.
MAX_STREAM_CHARS = 200_000

#: Called with each fragment, in arrival order. Fragments concatenate to the
#: full text; no fragment is a complete line, sentence or token boundary.
StreamSink = Callable[[str], None]

#: Polled between fragments. True means stop now. The half-written answer is
#: still returned — the user asked to stop reading, not to discard.
CancelCheck = Optional[Callable[[], bool]]


@dataclass(frozen=True)
class StreamResult:
    """What arrived, and whether that was all of it."""

    text: str
    #: True only when the format's own end-of-stream marker was seen. A socket
    #: that closed cleanly mid-answer is NOT complete.
    complete: bool
    #: Empty when the stream ended normally. Otherwise the stated reason —
    #: shown to the user, so it must be readable rather than a repr.
    error: str = ""
    #: The user pressed stop. Distinguished from `error` because it is not a
    #: failure and must not be presented as one.
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        """Usable as a whole answer: it finished and nothing went wrong."""
        return self.complete and not self.error and not self.cancelled


# ── the reasoning scratchpad, removed as it arrives ──────────────────────────
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _partial_tail(text: str, tag: str) -> int:
    """Length of the longest proper prefix of `tag` that ends `text`.

    A fragment boundary can fall anywhere, including the middle of `</think>`.
    Emitting that half and then the other half would print the tag; holding the
    whole buffer back would stall the panel. Holding back exactly this many
    characters does neither.
    """
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ThinkFilter:
    """Strip `<think>…</think>` from a stream, incrementally.

    Reasoning models narrate to themselves before answering. `engine._strip_think`
    removes that from a finished reply, and the live view has to remove exactly
    the same span or the user watches a scratchpad appear and then vanish when
    the answer is saved. This is that function's streaming twin, and it matches
    it in the awkward case too: an opening tag that is never closed swallows the
    rest, because the alternative is leaking the scratchpad.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, fragment: str) -> str:
        """Return the part of `fragment` that belongs in the transcript."""
        self._buffer += fragment or ""
        out: list[str] = []
        while True:
            if self._inside:
                at = self._buffer.find(_CLOSE_TAG)
                if at < 0:
                    keep = _partial_tail(self._buffer, _CLOSE_TAG)
                    self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
                    break
                self._buffer = self._buffer[at + len(_CLOSE_TAG):]
                self._inside = False
                continue
            at = self._buffer.find(_OPEN_TAG)
            if at < 0:
                keep = _partial_tail(self._buffer, _OPEN_TAG)
                if keep:
                    out.append(self._buffer[:len(self._buffer) - keep])
                    self._buffer = self._buffer[len(self._buffer) - keep:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break
            out.append(self._buffer[:at])
            self._buffer = self._buffer[at + len(_OPEN_TAG):]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        """Whatever was held back for a tag that never arrived."""
        if self._inside:
            self._buffer = ""
            return ""
        out, self._buffer = self._buffer, ""
        return out


# ── pure decoders ────────────────────────────────────────────────────────────
def decode_ollama_line(line: str) -> Tuple[str, bool]:
    """`(fragment, done)` for one NDJSON line of an Ollama chat stream.

    An unparseable line yields `("", False)` rather than raising. Ollama emits
    keep-alive blanks, and a strict parser here would turn one into a failed
    answer.
    """
    text = (line or "").strip()
    if not text:
        return "", False
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return "", False
    if not isinstance(obj, dict):
        return "", False
    done = bool(obj.get("done"))
    message = obj.get("message")
    fragment = ""
    if isinstance(message, dict):
        fragment = str(message.get("content") or "")
    if not fragment:
        # `/api/generate` shape, accepted because some deployments proxy it.
        fragment = str(obj.get("response") or "")
    return fragment, done


def sse_events(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """`(event, data)` pairs from Server-Sent Events lines.

    Follows the dispatch rule rather than assuming one `data:` per event: the
    payload accumulates across `data:` lines and is delivered on the blank line
    that closes the event. Anthropic sends `event:` before its data; the
    OpenAI-compatible format sends data alone, which arrives here as `("", …)`.
    """
    event = ""
    data: list[str] = []
    for raw in lines:
        line = (raw or "").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "", []
            continue
        if line.startswith(":"):
            continue                       # comment / keep-alive
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:                               # a stream that ended without a blank
        yield event, "\n".join(data)


def decode_openai_event(data: str) -> Tuple[str, bool]:
    """`(fragment, done)` for one OpenAI-compatible SSE payload."""
    payload = (data or "").strip()
    if not payload:
        return "", False
    if payload == "[DONE]":
        return "", True
    try:
        obj = json.loads(payload)
    except Exception:  # noqa: BLE001
        return "", False
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        return "", False
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
    content = delta.get("content")
    if isinstance(content, list):
        # Some backends type their deltas the way the non-streaming reply does.
        content = "".join(p.get("text", "") for p in content
                          if isinstance(p, dict) and p.get("type") in (None, "text"))
    # `finish_reason` closes the choice; `[DONE]` closes the stream. Either is
    # a legitimate end — several servers send only the former.
    done = bool(first.get("finish_reason"))
    return str(content or ""), done


def decode_anthropic_event(event: str, data: str) -> Tuple[str, bool]:
    """`(fragment, done)` for one Anthropic SSE event.

    `error` events are surfaced as a completion so the loop stops rather than
    waiting out the socket timeout on a stream the server has abandoned.
    """
    name = (event or "").strip()
    payload = (data or "").strip()
    obj = None
    if payload:
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001
            obj = None
    if not name and isinstance(obj, dict):
        name = str(obj.get("type") or "")
    if name in ("message_stop", "error"):
        return "", True
    if name != "content_block_delta" or not isinstance(obj, dict):
        return "", False
    delta = obj.get("delta")
    if not isinstance(delta, dict):
        return "", False
    if delta.get("type") not in (None, "text_delta"):
        return "", False               # thinking / tool-input deltas are not prose
    return str(delta.get("text") or ""), False


# ── the shared loop ──────────────────────────────────────────────────────────
def _pump(response, decode, sink: StreamSink,
          cancel: CancelCheck) -> StreamResult:
    """Drive one decoded-line iterator into the sink.

    `decode` maps a raw line to `(fragment, done)`. The accumulator is the
    return value; the sink is best-effort, and a sink that raises must not cost
    the text already received — the caller still has a complete answer to save.
    """
    parts: list[str] = []
    total = 0
    for raw in response:
        if cancel is not None and cancel():
            return StreamResult("".join(parts), False, cancelled=True)
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        fragment, done = decode(line)
        if fragment:
            parts.append(fragment)
            total += len(fragment)
            try:
                sink(fragment)
            except Exception:  # noqa: BLE001 — the consumer's problem, not the text's
                pass
            if total >= MAX_STREAM_CHARS:
                return StreamResult(
                    "".join(parts), False,
                    error=f"the answer passed {MAX_STREAM_CHARS:,} characters "
                          f"and was cut off")
        if done:
            return StreamResult("".join(parts), True)
    # The iterator ended without the format's end marker. That is a dropped
    # connection, not a short answer, and saying so is the point of this module.
    return StreamResult("".join(parts), False,
                        error="the model's connection closed before it finished")


def _sse_pump(response, decode_pair, sink: StreamSink,
              cancel: CancelCheck) -> StreamResult:
    """`_pump` for the two SSE formats, whose unit is an event, not a line."""
    parts: list[str] = []
    total = 0

    def _lines():
        for raw in response:
            if cancel is not None and cancel():
                return
            yield raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)

    for event, data in sse_events(_lines()):
        fragment, done = decode_pair(event, data)
        if fragment:
            parts.append(fragment)
            total += len(fragment)
            try:
                sink(fragment)
            except Exception:  # noqa: BLE001
                pass
            if total >= MAX_STREAM_CHARS:
                return StreamResult(
                    "".join(parts), False,
                    error=f"the answer passed {MAX_STREAM_CHARS:,} characters "
                          f"and was cut off")
        if done:
            return StreamResult("".join(parts), True)
    if cancel is not None and cancel():
        return StreamResult("".join(parts), False, cancelled=True)
    return StreamResult("".join(parts), False,
                        error="the model's connection closed before it finished")


def _open(url: str, body: dict, headers: dict, timeout: float):
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _failed(exc: Exception) -> StreamResult:
    return StreamResult("", False, error=f"the model could not be reached "
                                         f"({type(exc).__name__})")


# ── the three backends ───────────────────────────────────────────────────────
def ollama_stream(base_url: str = "http://localhost:11434",
                  model: str = "qwen3-coder:30b", *, temperature: float = 0.1,
                  timeout: float = 300.0, max_tokens: Optional[int] = None):
    """A streaming counterpart to `providers.ollama_llm`.

    Shares its URL normalisation deliberately: two functions deriving the same
    endpoint independently is how one of them ends up posting to
    `/api/chat/api/chat`.
    """
    from alelyon.runtime.oracle.answer.providers import normalize_ollama_chat_url

    endpoint = normalize_ollama_chat_url(base_url)

    def _stream(prompt: str, sink: StreamSink,
                cancel: CancelCheck = None) -> StreamResult:
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max(1, int(max_tokens))
        body = {"model": model, "stream": True, "options": options,
                "messages": [{"role": "user", "content": prompt}]}
        try:
            with _open(endpoint, body, {"Content-Type": "application/json"},
                       timeout) as response:
                return _pump(response, decode_ollama_line, sink, cancel)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)

    return _stream


def openai_compatible_stream(base_url: str, model: str, *, api_key: str = "",
                             api_key_name: str = "", temperature: float = 0.1,
                             timeout: float = 300.0,
                             max_tokens: Optional[int] = None,
                             organization: str = ""):
    """A streaming counterpart to `providers.openai_compatible_llm`.

    The key is resolved at call time for the same reason it is there: a key
    entered while the application is running should work without a restart, and
    it should not sit in a closure any longer than the request that uses it.
    """
    from alelyon.runtime.oracle.answer.providers import normalize_openai_chat_url

    endpoint = normalize_openai_chat_url(base_url)

    def _stream(prompt: str, sink: StreamSink,
                cancel: CancelCheck = None) -> StreamResult:
        if not endpoint:
            return StreamResult("", False, error="no endpoint is configured")
        key = api_key
        if not key and api_key_name:
            try:
                from alelyon.runtime.atlas.data.keys import get_key
                key = get_key(api_key_name) or ""
            except Exception:  # noqa: BLE001
                key = ""
        body = {"model": model, "stream": True, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if max_tokens is not None:
            body["max_tokens"] = max(1, int(max_tokens))
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if organization:
            headers["OpenAI-Organization"] = organization
        try:
            with _open(endpoint, body, headers, timeout) as response:
                return _sse_pump(response,
                                 lambda _event, data: decode_openai_event(data),
                                 sink, cancel)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)

    return _stream


def anthropic_stream(model: str = "claude-sonnet-4-5-20250929", *,
                     max_tokens: int = 1200, temperature: float = 0.1,
                     timeout: float = 90.0):
    """A streaming counterpart to `providers.anthropic_llm`.

    Reaches the network only with a configured key, and returns a stated reason
    rather than an empty answer when there is none — the caller has to be able
    to tell "no key" from "the model said nothing".
    """
    from alelyon.runtime.oracle.answer.providers import (
        _ANTHROPIC_URL, _ANTHROPIC_VERSION,
    )

    def _stream(prompt: str, sink: StreamSink,
                cancel: CancelCheck = None) -> StreamResult:
        try:
            from alelyon.runtime.atlas.data.keys import get_key
            key = get_key("ANTHROPIC_API_KEY")
        except Exception:  # noqa: BLE001
            key = None
        if not key:
            return StreamResult("", False, error="no ANTHROPIC_API_KEY is set")
        body = {"model": model, "max_tokens": max_tokens, "stream": True,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": _ANTHROPIC_VERSION}
        try:
            with _open(_ANTHROPIC_URL, body, headers, timeout) as response:
                return _sse_pump(response, decode_anthropic_event, sink, cancel)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)

    return _stream
