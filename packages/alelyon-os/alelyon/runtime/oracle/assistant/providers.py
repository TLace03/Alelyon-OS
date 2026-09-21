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

from alelyon.runtime.oracle.answer.chat import (
    ChatFn, ChatMessage, ChatReply, ChatStreamFn, anthropic_chat,
    anthropic_chat_stream, flatten_messages, ollama_chat, ollama_chat_stream,
    openai_chat, openai_chat_stream,
)
from alelyon.runtime.oracle.answer.providers import (
    anthropic_llm, have_anthropic_key, ollama_llm, openai_compatible_llm,
)
from alelyon.runtime.oracle.answer.streaming import (
    CancelCheck, StreamResult, StreamSink, anthropic_stream,
    ollama_stream, openai_compatible_stream,
)
from alelyon.runtime.oracle.assistant import local_model as LM

# Compatibility alias.  ``local_model`` owns both the shipped default and the
# persisted/env-selected runtime model; do not introduce a second literal here.
DEFAULT_OLLAMA_MODEL = LM.DEFAULT_MODEL


#: The first transformers release that does not execute attacker-selected
#: kernel code while loading a crafted config (CVE-2026-4372). Mirrors the
#: constraint pyproject's `train` extra already declares; kept here because a
#: source pin is a statement of intent and this is the code that would actually
#: do the loading.
_MIN_SAFE_TRANSFORMERS = (5, 3)

#: Generation budget for the legacy single-prompt in-process seam. 64 was too
#: short for an assistant answer and produced sentences that stop mid-clause;
#: the chat seam in `local_hf` has its own budget.
_HF_MAX_NEW_TOKENS = 512

#: Private/source compositions may supply the native in-process chat adapter.
#: The public provider module deliberately owns no import path to that adapter:
#: ``alelyon-os`` ships this file but not ``local_hf.py``.  Keeping the seam as
#: an injected callable makes the public artifact's legacy single-prompt
#: fallback the default while allowing the desktop source product to opt in.
HFChatFactory = Callable[..., tuple[ChatFn, ChatStreamFn]]


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


#: Where a chosen in-process model directory is remembered. Same reasoning as
#: `local_model._PREF_PATH`: the API server and any headless caller need this
#: too, so it cannot live in QSettings. Env still wins, so a machine can
#: override without touching stored state.
def _hf_pref_path() -> Path:
    from alelyon.runtime.common.paths import GLOBALS_DIR  # noqa: PLC0415

    return Path(GLOBALS_DIR) / "hf_model_dir.json"


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
    try:
        import json  # noqa: PLC0415

        raw = json.loads(_hf_pref_path().read_text(encoding="utf-8"))
        return str((raw or {}).get("model_dir", "") or "").strip()
    except Exception:  # noqa: BLE001 - an unreadable preference is "unset"
        return ""


def set_hf_model_dir(path: str) -> bool:
    """Remember an in-process model directory, or forget it when given "".

    Writes through a temp file so a crash mid-write cannot leave a half-parsed
    preference that reads as a DIFFERENT directory rather than as none.
    """
    import json  # noqa: PLC0415

    value = str(path or "").strip()
    pref = _hf_pref_path()
    try:
        pref.parent.mkdir(parents=True, exist_ok=True)
        tmp = pref.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model_dir": value}), encoding="utf-8")
        tmp.replace(pref)
    except Exception:  # noqa: BLE001
        return False
    # The answer to "can HF serve" changes with the directory, and both are
    # cached for the life of the process.
    hf_import_works.cache_clear()
    return True


def prepare_transformers_import() -> None:
    """Make `import transformers` survive a torch build with no distributed
    backend. Process-local; site-packages is never written.

    This machine's torch is `2.9.1+rocm7.2.1`, which reports
    `torch.distributed.is_available() == False` and ships no
    `torch._C._distributed_c10d`. Three upstream sites assume otherwise, and
    all three are import-time or load-time rather than "only if you use FSDP":

      * `transformers.distributed.fsdp` — guards its FSDP imports on torch
        VERSION (>= 2.6) instead of on `torch.distributed.is_available()`, so
        importing anything through `generation/utils.py` raises.
      * `transformers.distributed.sharding_utils` — unconditional
        `from torch.distributed.tensor import DTensor`, reached through
        `core_model_loading` on every model load.
      * `transformers.core_model_loading` — imports `DTensor` under
        `if torch.distributed.is_available():` and then USES the name
        unguarded (5.14.1, lines 1333/1343/1351/1662). On a build where that
        is False the name is never bound and EVERY load raises
        `NameError: name 'DTensor' is not defined`. That one is an upstream
        bug rather than an unsupported configuration: the guard and the use
        disagree.

    The stubs answer exactly what the real code answers when there is no
    distributed backend — FSDP off, and nothing is a DTensor — so this narrows
    behaviour to the true configuration rather than faking a capability. Any
    genuine FSDP/DTensor path raises rather than silently doing nothing.
    """
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    def _raiser(name: str):
        # Dunders MUST fall through: `inspect.getmodule` probes every module in
        # sys.modules for `__file__`, and answering with a function crashes any
        # caller expecting a str.
        if name.startswith("__"):
            raise AttributeError(name)

        def _fn(*_a, **_k):
            raise RuntimeError(
                f"{name} unavailable: this torch build has no distributed backend")

        _fn.__name__ = name
        return _fn

    try:
        import transformers.distributed.fsdp  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        stub = types.ModuleType("transformers.distributed.fsdp")
        stub.is_fsdp_enabled = lambda: False
        stub.is_fsdp_managed_module = lambda module: False
        stub.verify_fsdp_plan = _raiser("verify_fsdp_plan")
        stub.__getattr__ = _raiser
        sys.modules["transformers.distributed.fsdp"] = stub

    try:
        import transformers.distributed.sharding_utils  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        shard = types.ModuleType("transformers.distributed.sharding_utils")

        class DtensorShardOperation:  # noqa: N801 - mirrors upstream name
            # A real class, because it appears in a def-time `X | None`
            # annotation; instantiating it is the unsupported path.
            def __init__(self, *_a, **_k):
                raise RuntimeError(
                    "DTensor sharding unavailable: no distributed backend")

        shard.DtensorShardOperation = DtensorShardOperation
        shard._dtensor_from_local_like = _raiser("_dtensor_from_local_like")
        shard.__getattr__ = _raiser
        sys.modules["transformers.distributed.sharding_utils"] = shard


def bind_absent_dtensor() -> None:
    """Bind the name `core_model_loading` uses but may never have imported.

    Separate from `prepare_transformers_import` because it must run AFTER
    that module is imported, not before. With no distributed backend no tensor
    can BE a DTensor, so `isinstance(x, _AbsentDTensor)` is correctly False for
    every x — which is the answer the upstream branch wants and cannot reach.
    """
    import importlib  # noqa: PLC0415

    try:
        # Resolve through the import registry.  ``import package.child as x``
        # may reuse a stale ``package.child`` attribute after a test/plugin has
        # replaced the fully-qualified module in ``sys.modules``; patching that
        # detached object leaves the loader which will actually run unchanged.
        _cml = importlib.import_module("transformers.core_model_loading")
    except Exception:  # noqa: BLE001
        return
    if not hasattr(_cml, "DTensor"):
        class _AbsentDTensor:
            """Nothing is an instance of this, deliberately."""

        _cml.DTensor = _AbsentDTensor


@lru_cache(maxsize=1)
def hf_import_works() -> bool:
    """Can this process actually import the loader, shims applied?

    Cached: importing transformers costs seconds, and the answer cannot change
    within a process. Deliberately does NOT load weights — that is per-model,
    while this is per-installation.
    """
    if not transformers_load_is_safe():
        return False
    try:
        prepare_transformers_import()
        from transformers import (  # noqa: F401,PLC0415
            AutoModelForCausalLM, AutoTokenizer,
        )
        bind_absent_dtensor()
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("[assistant] in-process Hugging Face loading is not "
                     "available in this environment: %s: %s",
                     type(exc).__name__, exc)
        return False


@dataclass
class Provider:
    name: str                     # "ollama:qwen3-coder:30b" | "claude-sonnet-5"
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
    #: The conversation-native seams, when the backend has them. A backend with
    #: neither is still fully usable through `chat()`: the messages are
    #: rendered by `flatten_messages` — the one flattening — and the reply says
    #: `native=False`, because "the model read a rendition" is a fact the
    #: transcript must be able to state.
    chat_fn: Optional[ChatFn] = None
    chat_stream_fn: Optional[ChatStreamFn] = None

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

    @property
    def conversational(self) -> bool:
        """Whether this backend sees a message array natively. Reported so a
        surface can say which it is; `chat()` works either way."""
        return self.chat_fn is not None or self.chat_stream_fn is not None

    def chat(self, messages: "list[ChatMessage]",
             sink: Optional[StreamSink] = None,
             cancel: CancelCheck = None) -> ChatReply:
        """One conversational exchange, streamed when a sink is given.

        Four cases, all yielding the same shape:

        * native seam + sink → the backend streams the message array;
        * native seam, no sink → one native round trip;
        * no native seam + sink → `flatten_messages` through `stream()`, the
          whole reply as fragments, `native=False`;
        * no native seam, no sink → the flattening through the legacy
          callable, `native=False`.

        The fallback is not skipped and not an error, for the same reason
        `stream()` gives: the most private backend on the machine must not
        stop being usable the day the conversation went native.
        """
        if sink is not None and self.chat_stream_fn is not None:
            try:
                return self.chat_stream_fn(messages, sink, cancel)
            except Exception as exc:  # noqa: BLE001 — never raise into a GUI thread
                return ChatReply("", False, error=f"{type(exc).__name__}: {exc}")
        if self.chat_fn is not None:
            if cancel is not None and cancel():
                return ChatReply("", False, cancelled=True)
            try:
                reply = self.chat_fn(messages)
            except Exception as exc:  # noqa: BLE001
                return ChatReply("", False, error=f"{type(exc).__name__}: {exc}")
            if sink is not None and reply.text:
                try:
                    sink(reply.text)
                except Exception:  # noqa: BLE001
                    pass
            return reply
        prompt = flatten_messages(messages)
        if sink is not None:
            result = self.stream(prompt, sink, cancel)
            return ChatReply(result.text, result.complete, error=result.error,
                             cancelled=result.cancelled, native=False)
        if cancel is not None and cancel():
            return ChatReply("", False, cancelled=True, native=False)
        try:
            text = (self.fn(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001
            return ChatReply("", False, native=False,
                             error=f"{type(exc).__name__}: {exc}")
        if not text:
            return ChatReply("", False, native=False,
                             error="the model returned nothing")
        return ChatReply(text, True, native=False)


def ollama_provider(model: str = "", base_url: str = "") -> Provider:
    # An explicit argument describes a deliberately separate composition.  The
    # no-argument Local/Auto provider follows the same selection as ModelBar and
    # the built-in registry endpoint.
    model = str(model or "").strip() or LM.selected_model()
    base = str(base_url or "").strip() or LM.base_url()
    return Provider(name=f"ollama:{model}", fn=ollama_llm(base, model, temperature=0.2),
                    local=True, grammar=True,
                    stream_fn=ollama_stream(base, model, temperature=0.2),
                    chat_fn=ollama_chat(base, model, temperature=0.2),
                    chat_stream_fn=ollama_chat_stream(base, model, temperature=0.2))


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
        # Shims for a torch build with no distributed backend, applied before
        # the import and before any load. See `prepare_transformers_import`.
        try:
            prepare_transformers_import()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            bind_absent_dtensor()
        except Exception as exc:  # noqa: BLE001
            _log.warning("[assistant] could not import the Hugging Face "
                         "loader: %s: %s", type(exc).__name__, exc)
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=False)
            model.eval()
            return tokenizer, model
        except Exception as exc:  # noqa: BLE001
            # Named rather than swallowed: the caller's only other signal is an
            # empty answer, which is indistinguishable from a model that
            # declined to say anything.
            _log.warning("[assistant] could not load the Hugging Face model at "
                         "%s: %s: %s", model_path, type(exc).__name__, exc)
            return None

    def _fn(prompt: str, schema=None) -> str:
        if schema is not None:
            return ""
        loaded = _load()
        if loaded is None:
            return ""
        tokenizer, model = loaded
        try:
            import torch  # noqa: PLC0415

            # An instruct checkpoint answers its own chat template and rambles
            # without it — measured: this path returned the prompt back plus a
            # invented article title. Fall back to the raw prompt only when the
            # tokenizer declares no template.
            text = prompt
            if getattr(tokenizer, "chat_template", None):
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)
                except Exception:  # noqa: BLE001
                    text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=_HF_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None))
            # Decode ONLY what was generated. Decoding outputs[0] whole returns
            # the prompt with the answer glued to it, which every caller then
            # has to strip and none of them did.
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            _log.warning("[assistant] in-process generation failed: %s: %s",
                         type(exc).__name__, exc)
            return ""

    return _fn


def _hf_chat_fns(
        resolved_dir: str,
        hf_chat_factory: Optional[HFChatFactory] = None,
) -> tuple[Optional[ChatFn], Optional[ChatStreamFn]]:
    """Build the private native-chat seam only when composition supplies it.

    ``providers.py`` ships in the public wheel while the hardened adapter does
    not.  Discovering that adapter from here made the source checkout's import
    closure depend on private files and on the operator's persisted model
    preference.  Injection keeps the boundary structural: no factory means the
    documented legacy single-prompt seam, without attempting a private import.
    The CVE gate is passed in so the rule still lives exactly once, here.
    """
    if not resolved_dir or hf_chat_factory is None:
        return None, None
    try:
        return hf_chat_factory(
            resolved_dir,
            safe=transformers_load_is_safe,
            refusal_reason=_transformers_refusal_reason,
        )
    except Exception as exc:  # noqa: BLE001 - adapter refusal stays optional
        _log.warning(
            "[assistant] native local chat adapter was not composed: %s",
            type(exc).__name__,
        )
        return None, None


def _label_for_model_dir(resolved_dir: str) -> str:
    """A name a reader recognises, not the directory that happened to hold it.

    The provider's name travels with the answer into the saved transcript (see
    this module's docstring), and a Hugging Face cache directory is
    `…/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/<40 hex chars>`,
    whose `.name` is the hash. `hf:ea3f2471cf1b…` tells a reader in three
    months nothing at all — not even which model family answered — so walk up
    to the `models--Org--Name` component and turn it back into `Org/Name`.
    Any other layout keeps its own directory name, which for a hand-managed
    checkout is already the informative part.
    """
    path = Path(resolved_dir)
    for part in (path, *path.parents):
        if part.name.startswith("models--"):
            return part.name[len("models--"):].replace("--", "/", 1)
    return path.name


def hf_provider(
        model_dir: str = "",
        model: str = "",
        *,
        hf_chat_factory: Optional[HFChatFactory] = None,
) -> Provider:
    resolved_dir = str(model_dir or _configured_hf_model_dir() or "").strip()
    if resolved_dir:
        label = _label_for_model_dir(resolved_dir) or str(model or "local-hf")
    else:
        label = str(model or "local-hf")
    chat_fn, chat_stream_fn = _hf_chat_fns(resolved_dir, hf_chat_factory)
    return Provider(name=f"hf:{label}", fn=hf_llm(resolved_dir), local=True,
                    grammar=False, chat_fn=chat_fn,
                    chat_stream_fn=chat_stream_fn)


def hf_is_usable() -> bool:
    """Is the in-process Hugging Face path both configured AND able to load?

    All THREE halves matter. A configured model directory on a machine whose
    transformers cannot safely load it yields a provider that returns "" to
    every request — and a provider that structurally cannot answer does not
    belong at the front of the chain, or anywhere in it. Offering it would be
    exactly the "availability is not correctness" mistake: the list looks
    richer and every answer still comes from the next entry down.

    Until 2026-08-21 this asked only whether a directory was configured and
    whether the CVE gate allowed loading, and called that "able to load". It
    was not: on this workstation's torch build `import transformers` itself
    raised, so the honest answer was False while this returned True. That is
    not hypothetical — `chat_workspace` builds `Chain([default_local_provider()])`
    with NO second provider, so the failure would have surfaced as a chat that
    answers every question with silence. `hf_import_works()` closes it.

    Order is deliberate. The cheap checks come first and short-circuit, so a
    machine that has configured nothing — everyone, by default — never pays the
    import. Only an operator who explicitly asked for in-process serving pays
    it, once per process, and paying seconds to learn whether the thing they
    asked for actually works is the correct trade.
    """
    return (bool(_configured_hf_model_dir())
            and transformers_load_is_safe()
            and hf_import_works())


def default_local_provider(
        *, hf_chat_factory: Optional[HFChatFactory] = None) -> Provider:
    """Select the configured local backend, preferring Hugging Face when it is
    configured AND loadable."""
    if hf_is_usable():
        return hf_provider(hf_chat_factory=hf_chat_factory)
    return ollama_provider()


def anthropic_provider(model: str = "") -> Provider:
    model = model or os.environ.get("ANALYST_CLOUD_MODEL") or "claude-sonnet-5"
    return Provider(name=model, fn=anthropic_llm(model), local=False, grammar=False,
                    stream_fn=anthropic_stream(model),
                    chat_fn=anthropic_chat(model),
                    chat_stream_fn=anthropic_chat_stream(model))


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

    endpoint = MC.runtime_endpoint(endpoint)
    if endpoint.kind == MC.KIND_ANTHROPIC:
        key_name = endpoint.api_key_name or "ANTHROPIC_API_KEY"
        fn = anthropic_llm(endpoint.model)
        stream_fn = anthropic_stream(endpoint.model)
        chat_fn = anthropic_chat(endpoint.model, api_key_name=key_name)
        chat_stream_fn = anthropic_chat_stream(endpoint.model,
                                               api_key_name=key_name)
        grammar = False
    elif endpoint.kind == MC.KIND_OLLAMA:
        base = (endpoint.base_url or os.environ.get("OLLAMA_BASE_URL")
                or "http://localhost:11434")
        fn = ollama_llm(base, endpoint.model, temperature=0.2)
        stream_fn = ollama_stream(base, endpoint.model, temperature=0.2)
        chat_fn = ollama_chat(base, endpoint.model, temperature=0.2)
        chat_stream_fn = ollama_chat_stream(base, endpoint.model,
                                            temperature=0.2)
        grammar = True
    else:
        fn = openai_compatible_llm(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        stream_fn = openai_compatible_stream(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        chat_fn = openai_chat(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        chat_stream_fn = openai_chat_stream(
            endpoint.base_url, endpoint.model,
            api_key_name=endpoint.api_key_name, temperature=0.2)
        grammar = False
    return Provider(name=f"{endpoint.id}:{endpoint.model}", fn=fn,
                    local=bool(endpoint.local), grammar=grammar,
                    stream_fn=stream_fn, chat_fn=chat_fn,
                    chat_stream_fn=chat_stream_fn)


def available(*, hf_chat_factory: Optional[HFChatFactory] = None) -> List[Provider]:
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
        out.append(hf_provider(hf_chat_factory=hf_chat_factory))

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
        # ON by default — the safe state is the default state (RT15-09). A
        # default-built chain that is marked private keeps the question on
        # this machine; a caller for whom cloud egress is the user's explicit,
        # visible choice opts OUT, and the opt-out is greppable at the site
        # that made the promise.
        local_only_after_private: bool = True,
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

    def chat(self, messages: "list[ChatMessage]",
             sink: Optional[StreamSink] = None,
             cancel: CancelCheck = None) -> ChatReply:
        """One conversational exchange from the first provider that answers.

        The fallback rule matches `stream()`, not `__call__`, and for the same
        reason: with a sink, fragments may already be on the user's screen, so
        the chain falls through only while nothing has been emitted. Without a
        sink it may try every candidate, exactly as `__call__` does — nobody
        has seen the failed attempts.

        The privacy filter applies before the first candidate is tried, in
        both shapes. A cancellation stops the chain outright.
        """
        self.attempts = []
        candidates = [self._selected] if self._sticky and self._selected else self.providers
        if self._private and self._local_only_after_private:
            candidates = [p for p in candidates if p.local]
        last = ChatReply("", False, error="no provider was available")
        for provider in candidates:
            if cancel is not None and cancel():
                return ChatReply("", False, cancelled=True)
            reply = provider.chat(messages, sink, cancel)
            self.attempts.append(provider.name)
            if reply.cancelled:
                self.used = provider.name if reply.text else self.used
                return reply
            if reply.text:
                self.used = provider.name
                if self._sticky and self._selected is None:
                    self._selected = provider
                return reply
            last = reply
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
