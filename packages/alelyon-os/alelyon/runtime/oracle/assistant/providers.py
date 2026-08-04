"""Which model answers, and the fallback when it cannot.

The audit's third AI-Analyst gap: the panel hard-coded `http://localhost:11434`
and the model name `quantmaster`, so with Ollama down the feature was simply
dead, and there was no way to reach a stronger model for a hard question. The
guidance extractor already had a provider seam; this reuses it.

The provider's NAME travels with the answer all the way into the saved
transcript. Two models will answer the same question differently, and a thread
read back in three months that does not say which one wrote it is missing the
first thing you would want to know.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional

_log = logging.getLogger(__name__)

from alelyon.runtime.oracle.answer.providers import (
    anthropic_llm, have_anthropic_key, ollama_llm, openai_compatible_llm,
)
from alelyon.runtime.oracle.answer.streaming import (
    CancelCheck, StreamResult, StreamSink, anthropic_stream,
    ollama_stream, openai_compatible_stream,
)

DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"


#: The first transformers release that does not execute attacker-selected
#: kernel code while loading a crafted config (CVE-2026-4372). Mirrors the
#: constraint pyproject's `train` extra already declares; kept here because a
#: source pin is a statement of intent and this is the code that would actually
#: do the loading.
_MIN_SAFE_TRANSFORMERS = (5, 3)


def _transformers_version() -> Optional[tuple]:
    """Installed transformers as a comparable tuple, or None if absent."""
    try:
        import importlib.metadata as _md

        raw = _md.version("transformers")
    except Exception:  # noqa: BLE001
        return None
    parts = []
    for chunk in str(raw).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def transformers_load_is_safe() -> bool:
    """May this process load a Hugging Face model in-process?

    False when transformers is absent or older than the fixed release. Absent
    counts as unsafe because the caller cannot load anything anyway, and
    answering True would put the decision in the exception handler.
    """
    version = _transformers_version()
    return bool(version and version >= _MIN_SAFE_TRANSFORMERS)


def _transformers_refusal_reason() -> str:
    version = _transformers_version()
    if version is None:
        return "transformers is not installed"
    shown = ".".join(str(p) for p in version)
    needed = ".".join(str(p) for p in _MIN_SAFE_TRANSFORMERS)
    return (f"transformers {shown} is installed; {needed}+ is required "
            f"(CVE-2026-4372)")


def _configured_hf_model_dir() -> str:
    for env_name in (
        "ALELYON_HF_MODEL_DIR",
        "ALELYON_HF_MODEL_PATH",
        "HF_MODEL_DIR",
        "HF_MODEL_PATH",
        "MODEL_DIRECTORY",
    ):
        value = os.environ.get(env_name)
        if value:
            return str(value).strip()
    return ""


@dataclass
class Provider:
    name: str                     # "ollama:qwen3-coder:30b" | "claude-sonnet-4-5…"
    fn: Callable[[str], str]
    local: bool
    #: Whether this backend can CONSTRAIN sampling to a JSON Schema.
    #:
    #: Declared per provider, not inferred. It used to be `return self.local`,
    #: on the reasoning that local meant Ollama and Ollama compiles a schema
    #: into a llama.cpp GBNF grammar. That stopped being true the moment a
    #: second local backend existed: an in-process Hugging Face loader and an
    #: OpenAI-compatible server are both local and neither restricts the
    #: sampler. `assistant.constrain` is built on the model being UNABLE to emit
    #: anything off-shape, so a provider that merely asks politely must not
    #: claim this — the answer would be badged as guaranteed when it is not.
    grammar: bool = False
    #: The backend's incremental seam, when it has one. Absent is normal and is
    #: not a defect: `stream()` below falls back to one blocking call delivered
    #: as a single fragment, so every caller sees the same shape and a provider
    #: without the seam is merely less pleasant to watch.
    stream_fn: Optional[Callable[..., StreamResult]] = None

    def __call__(self, prompt: str, schema=None) -> str:
        # A provider without a grammar seam must REFUSE a constrained request,
        # not silently answer it unconstrained — the caller would badge the
        # result as guaranteed.
        if schema is not None:
            if not self.grammar:
                return ""
            try:
                return self.fn(prompt, schema=schema)
            except TypeError:
                return ""
        return self.fn(prompt)

    @property
    def supports_grammar(self) -> bool:
        return self.grammar

    @property
    def incremental(self) -> bool:
        """Does this backend deliver fragments, or only a finished answer?

        Reported so a caller can say which it is. `stream()` works either way;
        what differs is whether the panel fills in gradually or all at once, and
        a progress line that claims streaming on a provider that cannot is the
        same lie as a badge claiming a grammar that was never applied.
        """
        return self.stream_fn is not None

    def stream(self, prompt: str, sink: StreamSink,
               cancel: CancelCheck = None) -> StreamResult:
        """Generate incrementally where the backend allows it.

        A provider with no streaming seam is not an error and is not skipped:
        it answers the ordinary way and its whole reply is delivered as one
        fragment. Skipping it would mean an in-process Hugging Face model — the
        most private option on the machine — silently stopped being usable the
        day the conversation started streaming.
        """
        if self.stream_fn is not None:
            try:
                return self.stream_fn(prompt, sink, cancel)
            except Exception as exc:  # noqa: BLE001 — never raise into a GUI thread
                return StreamResult("", False,
                                    error=f"{type(exc).__name__}: {exc}")
        if cancel is not None and cancel():
            return StreamResult("", False, cancelled=True)
        try:
            text = (self.fn(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001
            return StreamResult("", False, error=f"{type(exc).__name__}: {exc}")
        if not text:
            return StreamResult("", False, error="the model returned nothing")
        try:
            sink(text)
        except Exception:  # noqa: BLE001
            pass
        return StreamResult(text, True)


def ollama_provider(model: str = "", base_url: str = "") -> Provider:
    model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    return Provider(name=f"ollama:{model}", fn=ollama_llm(base, model, temperature=0.2),
                    local=True, grammar=True,
                    stream_fn=ollama_stream(base, model, temperature=0.2))


def hf_llm(model_dir: str = "", *, temperature: float = 0.1,
           timeout: float = 30.0) -> Callable[[str], str]:
    """A local Hugging Face fallback for the assistant provider chain.

    The runtime prefers this provider when a local model directory is configured
    through an environment variable. When the dependency stack or weights are
    unavailable, the callable returns an empty string so the chain can fail
    honestly rather than pretend to have answered.
    """
    resolved_dir = str(model_dir or _configured_hf_model_dir() or "").strip()
    if not resolved_dir:
        def _empty(_prompt: str, schema=None) -> str:
            return ""
        return _empty

    model_path = Path(resolved_dir).expanduser()

    @lru_cache(maxsize=1)
    def _load() -> Optional[tuple[object, object]]:
        if not transformers_load_is_safe():
            # Refuse, loudly enough to find in a log and quietly enough not to
            # break a GUI thread. pyproject's `train` extra requires
            # transformers>=5.3 because earlier releases execute
            # attacker-selected kernel code while loading a crafted config EVEN
            # WITH trust_remote_code=False (CVE-2026-4372) — so passing that
            # flag is not the mitigation it looks like. The documented rule is
            # that such a machine must refuse HF model loading, not proceed.
            _log.warning(
                "[assistant] refusing to load a Hugging Face model: %s. "
                "Serve the weights through a local OpenAI-compatible server "
                "(vLLM, TGI, llama.cpp, LM Studio) and add it in the model "
                "settings instead.", _transformers_refusal_reason())
            return None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:  # noqa: BLE001
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=False)
            model.eval()
            return tokenizer, model
        except Exception:  # noqa: BLE001
            return None

    def _fn(prompt: str, schema=None) -> str:
        if schema is not None:
            return ""
        loaded = _load()
        if loaded is None:
            return ""
        tokenizer, model = loaded
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with __import__("torch").no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            return ""

    return _fn


def hf_provider(model_dir: str = "", model: str = "") -> Provider:
    resolved_dir = str(model_dir or _configured_hf_model_dir() or "").strip()
    if resolved_dir:
        label = Path(resolved_dir).name or str(model or "local-hf")
    else:
        label = str(model or "local-hf")
    return Provider(name=f"hf:{label}", fn=hf_llm(resolved_dir), local=True,
                    grammar=False)


def hf_is_usable() -> bool:
    """Is the in-process Hugging Face path both configured AND able to load?

    Both halves matter. A configured model directory on a machine whose
    transformers cannot safely load it yields a provider that returns "" to
    every request — and a provider that structurally cannot answer does not
    belong at the front of the chain, or anywhere in it. Offering it would be
    exactly the "availability is not correctness" mistake: the list looks
    richer and every answer still comes from the next entry down.
    """
    return bool(_configured_hf_model_dir()) and transformers_load_is_safe()


def default_local_provider() -> Provider:
    """Select the configured local backend, preferring Hugging Face when it is
    configured AND loadable."""
    if hf_is_usable():
        return hf_provider()
    return ollama_provider()


def anthropic_provider(model: str = "") -> Provider:
    model = model or os.environ.get("ANALYST_CLOUD_MODEL") or "claude-sonnet-4-5-20250929"
    return Provider(name=model, fn=anthropic_llm(model), local=False, grammar=False,
                    stream_fn=anthropic_stream(model))


def endpoint_provider(endpoint) -> Provider:
    """Build a `Provider` from a `model_config.ModelEndpoint`.

    `local` comes from the endpoint's URL-derived property, never from its
    label. `Chain.mark_private()` trusts that flag to keep desk and book context
    off a machine that is not this one; deriving it from a user-supplied name
    would let an endpoint called "Local Qwen" point anywhere.

    Only Ollama advertises a grammar: it compiles a JSON Schema into a llama.cpp
    GBNF sampler grammar. An OpenAI-compatible server constrains by
    `response_format`, which is a request, not a sampler restriction.
    """
    from alelyon.runtime.oracle.assistant import model_config as MC

    if endpoint.kind == MC.KIND_ANTHROPIC:
        fn = anthropic_llm(endpoint.model)
        stream_fn = anthropic_stream(endpoint.model)
        grammar = False
    elif endpoint.kind == MC.KIND_OLLAMA:
        base = (endpoint.base_url or os.environ.get("OLLAMA_BASE_URL")
                or "http://localhost:11434")
        fn = ollama_llm(base, endpoint.model, temperature=0.2)
        stream_fn = ollama_stream(base, endpoint.model, temperature=0.2)
        grammar = True
    else:
        fn = openai_compatible_llm(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        stream_fn = openai_compatible_stream(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        grammar = False
    return Provider(name=f"{endpoint.id}:{endpoint.model}", fn=fn,
                    local=bool(endpoint.local), grammar=grammar,
                    stream_fn=stream_fn)


def available() -> List[Provider]:
    """Local first — it is free, private, and the book is on this machine. The
    cloud model is offered, never silently preferred.

    Reads the user's endpoint registry when anything in it is ready, so a
    fine-tuned model served locally, or a frontier key entered in the UI, takes
    effect. When the registry has nothing ready this falls back to the original
    pair, so a machine that has never opened the settings behaves exactly as it
    did before.
    """
    out: List[Provider] = []

    # An explicitly configured in-process model directory is a deliberate
    # operator override and stays ahead of the registry — but only when it can
    # actually load. See `hf_is_usable`.
    if hf_is_usable():
        out.append(hf_provider())

    try:
        from alelyon.runtime.oracle.assistant import model_config as MC

        configured = MC.ready_endpoints()
        if configured:
            return out + [endpoint_provider(e) for e in configured]
    except Exception:  # noqa: BLE001 - configuration never breaks the assistant
        pass

    out.append(ollama_provider())
    if have_anthropic_key():
        out.append(anthropic_provider())
    return out


class Chain:
    """Try each provider in turn; report which one actually answered.

    An empty string is a failure, not an answer — both providers return "" on a
    network error by contract, so a silent fall-through would otherwise look
    like the model choosing to say nothing.
    """

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        *,
        sticky: bool = False,
        local_only_after_private: bool = False,
    ):
        self.providers = list(providers) if providers is not None else available()
        self.used: str = ""
        self.attempts: List[str] = []
        self._sticky = bool(sticky)
        self._local_only_after_private = bool(local_only_after_private)
        self._private = False
        self._selected: Optional[Provider] = None

    def mark_private(self) -> None:
        """Prevent an Auto chain from spilling desk/book context to cloud.

        This is intentionally one-way for the lifetime of a question.  If a
        cloud provider already routed a public question, it does not gain
        authority to see the private facts returned afterwards; the engine can
        render those facts without model narration.
        """
        self._private = True

    @property
    def supports_grammar(self) -> bool:
        return any(p.supports_grammar for p in self.providers)

    def __call__(self, prompt: str, schema=None) -> str:
        self.attempts = []
        candidates = [self._selected] if self._sticky and self._selected else self.providers
        if self._private and self._local_only_after_private:
            candidates = [p for p in candidates if p.local]
        for p in candidates:
            if schema is not None and not p.supports_grammar:
                continue          # cannot enforce; do not pretend to
            text = ""
            try:
                text = p(prompt, schema=schema) if schema is not None else p(prompt)
                text = text or ""
            except Exception:  # noqa: BLE001
                text = ""
            self.attempts.append(p.name)
            if text.strip():
                self.used = p.name
                if self._sticky and self._selected is None:
                    self._selected = p
                return text
        if not self._sticky:
            self.used = ""
        return ""

    def stream(self, prompt: str, sink: StreamSink,
               cancel: CancelCheck = None) -> StreamResult:
        """Stream from the first provider that produces anything.

        The fallback rule is deliberately narrower here than in `__call__`, and
        the difference is the whole design of this method. `__call__` may try
        every provider because nobody has seen the failed attempts. A stream has
        already put text on the user's screen.

        So: fall through only while **nothing has been emitted**. Once a
        fragment has been delivered, that provider owns the answer — a failure
        after that point returns the partial text with `complete=False`, and the
        caller says so. Retrying with the next model would concatenate two
        models' prose into one answer under one provider name, which is a worse
        outcome than a visibly truncated one and is invisible in the transcript.

        A cancellation stops the chain outright. The user asked to stop, not to
        try somebody else.
        """
        self.attempts = []
        candidates = [self._selected] if self._sticky and self._selected else self.providers
        if self._private and self._local_only_after_private:
            candidates = [p for p in candidates if p.local]
        last = StreamResult("", False, error="no provider was available")
        for provider in candidates:
            if cancel is not None and cancel():
                return StreamResult("", False, cancelled=True)
            result = provider.stream(prompt, sink, cancel)
            self.attempts.append(provider.name)
            if result.cancelled:
                self.used = provider.name if result.text else self.used
                return result
            if result.text:
                self.used = provider.name
                if self._sticky and self._selected is None:
                    self._selected = provider
                return result
            last = result
        if not self._sticky:
            self.used = ""
        return last

    @property
    def failure_note(self) -> str:
        if self._private and self._local_only_after_private and not self.used:
            return ("the local model did not answer; cloud fallback was not "
                    "attempted because this question contains private desk or "
                    "book context")
        if self.used or not self.attempts:
            return ""
        tried = ", ".join(self.attempts)
        return (f"no model answered (tried: {tried}). Start the configured "
                f"local model, or set ANTHROPIC_API_KEY for the cloud fallback.")
