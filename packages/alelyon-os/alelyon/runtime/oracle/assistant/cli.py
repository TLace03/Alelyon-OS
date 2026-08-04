"""Workspace — the conversation, at a terminal.

    alelyon-workspace                       # start talking
    alelyon-workspace ask "what is 17% of 4210?"
    alelyon-workspace threads               # what you have asked before
    alelyon-workspace resume <id>
    alelyon-workspace models                # what can answer, and from where

This is the same assistant the desktop Workspace panel runs, driven from a
terminal instead of a Qt widget. Not a reimplementation: the panel and this both
call `engine.ask` over the `GENERAL` domain, so an answer here and an answer
there come from one engine, one tool layer and one grounding check. A second
implementation would drift, and the first thing to drift would be which figures
are claimed to be checked.

What you get, and what each part is worth
-----------------------------------------
**The model may reason.** This runs `MODE_OPEN`. The grounded contract — which
compiles the tools' rendered strings into a decoding grammar and forbids prose
from containing a digit — is the right rule beside a live book and the wrong one
for an assistant; it turns the model into a renderer of somebody else's table.
Here it may calculate, explain, write code, and answer from what it knows.

**The grounding check still runs, and is shown as provenance rather than as a
verdict.** Under each answer, figures that came from a deterministic tool are
separated from figures that are the model's own. That distinction is the point
of the product and it survives at a terminal: a number with a source beside a
number without one, never blended.

**The certified calculator is a real tool.** Ask for arithmetic over data and
the model authors a restricted program, the interpreter executes it, and the
result carries an error bar. The model does not compute the number.

**Nothing here places, cancels, or modifies anything.** The tool layer refuses
to register a tool whose name reads like an action, so the assistant cannot grow
a side effect by accident.

Visual design is deliberately plain for now
-------------------------------------------
ANSI colour where the stream is a terminal, nothing where it is not, and no
dependency on a TUI toolkit. The panel this mirrors has a session list, a
transcript that renders code as code, and a composer that takes more than one
line; this has the same shape at a much lower resolution, and it is meant to be
replaced by a real interface rather than admired.

Streaming, and why it is not optional
--------------------------------------
A local 30B model takes tens of seconds to compose a paragraph, and for all of
that time a prompt showing nothing is indistinguishable from one that has hung.
Fragments are written as they arrive. Ctrl-C during generation stops that answer
and keeps what arrived, labelled as stopped — a stream cut short and one the
reader ended are different events, and neither is a whole answer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from alelyon.runtime.oracle.assistant import history as H
from alelyon.runtime.oracle.assistant import providers as P
from alelyon.runtime.oracle.assistant import tools as T
from alelyon.runtime.oracle.assistant.domain import GENERAL, get_domain
from alelyon.runtime.oracle.assistant.engine import MODE_OPEN, ask

#: The conversation store. Separate from any other product's chat history: a
#: general assistant and a product-embedded one are different conversations and
#: merging them would put a stranger's context in front of either.
STORE = "workspace"

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREEN = "\033[32m"


def _colour_ok(stream) -> bool:
    """Colour only where it will be read as colour.

    `NO_COLOR` is honoured because it is the convention, and a pipe gets none
    because escape codes in a redirected transcript are noise a reader has to
    strip before the file is usable.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ALELYON_WORKSPACE_COLOR") == "0":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class _Paint:
    """Colour helpers that become the identity function when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.enabled else text

    def dim(self, text: str) -> str:
        return self(text, _DIM)

    def bold(self, text: str) -> str:
        return self(text, _BOLD)

    def cyan(self, text: str) -> str:
        return self(text, _CYAN)

    def warn(self, text: str) -> str:
        return self(text, _YELLOW)

    def bad(self, text: str) -> str:
        return self(text, _RED)

    def good(self, text: str) -> str:
        return self(text, _GREEN)


def _store_root() -> None:
    """Point the named store at its own directory under the state home.

    Registered here rather than at import so that importing this module has no
    filesystem effect — a CLI that creates directories on import cannot be
    imported by a test that meant only to read its parser.
    """
    from alelyon.runtime.common.paths import GLOBALS_DIR

    H.register_store(STORE, GLOBALS_DIR / "workspace_chat")


# ── providers ───────────────────────────────────────────────────────────────


def _resolve_provider(name: str) -> Optional[P.Provider]:
    """The named provider, or the first available one.

    A name that matches nothing REFUSES rather than falling back silently:
    asking for a specific model and quietly getting another one is how an
    answer gets attributed to a model that never saw the question.
    """
    options = P.available()
    if not name:
        return options[0] if options else None
    for provider in options:
        if provider.name == name or provider.name.startswith(name):
            return provider
    return None


def _describe_provider(provider: P.Provider, paint: _Paint) -> str:
    where = "local" if provider.local else "REMOTE — the question leaves this machine"
    bits = [where]
    bits.append("constrains sampling" if provider.grammar
                else "cannot constrain sampling")
    bits.append("streams" if provider.incremental else "answers all at once")
    return f"{paint.bold(provider.name)}  {paint.dim('(' + '; '.join(bits) + ')')}"


def _cmd_models(args, paint: _Paint) -> int:
    options = P.available()
    if not options:
        print("Nothing can answer.")
        print()
        print("Workspace needs a model. The usual local option is Ollama:")
        print("    ollama serve")
        print("    ollama pull qwen2.5:7b")
        print("and then set OLLAMA_MODEL if you want a different one.")
        return 1
    print(f"{len(options)} provider(s), in the order they would be tried:")
    print()
    for index, provider in enumerate(options, 1):
        print(f"  {index}. {_describe_provider(provider, paint)}")
    print()
    print(paint.dim(
        "Local first: it is free and the question stays on this machine. A "
        "remote\nprovider is offered, never silently preferred."))
    return 0


# ── threads ─────────────────────────────────────────────────────────────────


def _cmd_threads(args, paint: _Paint) -> int:
    _store_root()
    threads = H.list_threads(STORE)
    if not threads:
        print("No conversations yet.")
        return 0
    print(f"{len(threads)} conversation(s):")
    print()
    for thread in threads:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(thread.updated))
        print(f"  {paint.cyan(thread.id)}  {when}  "
              f"{thread.turns:>3} turns  {thread.title}")
    print()
    print(paint.dim(f"alelyon-workspace resume <id>"))
    return 0


# ── the answer, rendered ────────────────────────────────────────────────────


def _print_provenance(answer, paint: _Paint) -> None:
    """Separate what a tool returned from what the model wrote.

    This is the product, so it prints even when it is empty — "no tool was
    consulted" is information, and an absent section reads as an unremarkable
    answer rather than an unsourced one.
    """
    facts = answer.facts
    print()
    if facts:
        print(paint.dim("  ── from tools ────────────────────────────────────"))
        for fact in facts[:12]:
            label = getattr(fact, "label", "") or getattr(fact, "name", "")
            value = getattr(fact, "rendered", None) or getattr(fact, "value", "")
            asof = getattr(fact, "as_of", "") or ""
            stamp = paint.dim(f"  as of {asof}") if asof else ""
            print(f"  {paint.good('•')} {label}: {value}{stamp}")
        if len(facts) > 12:
            print(paint.dim(f"    (+{len(facts) - 12} more)"))
        if answer.consulted:
            print(paint.dim(f"    consulted: {answer.consulted}"))
    else:
        print(paint.dim("  ── from tools ────────────────────────────────────"))
        print(paint.dim("  nothing. No deterministic tool answered this, so every "
                        "figure above is the model's own."))

    unsupported = answer.unsupported_text
    if unsupported:
        print()
        print(paint.warn("  ── the model's own figures ───────────────────────"))
        for text in unsupported[:8]:
            print(f"  {paint.warn('?')} {text}")
        print(paint.dim(
            "  These trace to no tool. In this mode that is allowed and is not "
            "a defect —\n  it is the difference between a sourced number and an "
            "unsourced one, shown."))

    marks = []
    if answer.constrained:
        marks.append("decoder-constrained")
    if answer.deterministic:
        marks.append("routed without a model")
    if answer.cancelled:
        marks.append(paint.warn("STOPPED — this is not a whole answer"))
    if answer.truncated:
        marks.append(paint.warn("TRUNCATED — the model did not finish"))
    if answer.provider:
        marks.append(f"answered by {answer.provider}")
    if marks:
        print()
        print(paint.dim("  " + " · ".join(marks)))


def _answer(question: str, *, provider: P.Provider, thread_id: str,
            paint: _Paint, ctx: T.Context, domain) -> Optional[object]:
    """Ask once, streaming to stdout, and persist both turns."""
    H.append(thread_id, H.Turn(id=H._new_id(), ts=time.time(),
                               role=H.ROLE_USER, text=question), STORE)

    started = time.time()
    printed = {"any": False}

    def on_text(fragment: str) -> None:
        if not printed["any"]:
            printed["any"] = True
        sys.stdout.write(fragment)
        sys.stdout.flush()

    stop = {"now": False}

    def cancel() -> bool:
        return stop["now"]

    print()
    print(paint.cyan("Lattice"), end="  ", flush=True)
    print()
    try:
        answer = ask(
            question,
            ctx=ctx,
            llm=provider,
            history=H.recent_exchanges(thread_id, store=STORE),
            provider_name=provider.name,
            mode=MODE_OPEN,
            domain=domain,
            on_text=on_text,
            cancel=cancel,
        )
    except KeyboardInterrupt:
        stop["now"] = True
        print()
        print(paint.warn("  stopped. What arrived above is kept and is labelled "
                         "as stopped."))
        return None

    if not printed["any"] and answer.prose:
        # A provider with no streaming seam delivers the whole answer at once.
        print(answer.prose)
    elif printed["any"]:
        print()

    if answer.error:
        print(paint.bad(f"  {answer.error}"))

    _print_provenance(answer, paint)
    elapsed = time.time() - started
    print(paint.dim(f"  {elapsed:.1f}s"))

    H.append(thread_id, H.Turn(
        id=H._new_id(), ts=time.time(), role=H.ROLE_ASSISTANT,
        text=answer.prose,
        tools=list(answer.tools_run),
        unsupported=list(answer.unsupported_text),
        provider=answer.provider,
        error=answer.error,
        constrained=bool(answer.constrained),
        truncated=bool(answer.truncated),
        cancelled=bool(answer.cancelled),
    ), STORE)
    return answer


# ── one-shot ────────────────────────────────────────────────────────────────


def _cmd_ask(args, paint: _Paint) -> int:
    _store_root()
    provider = _resolve_provider(args.model)
    if provider is None:
        return _no_provider(args.model, paint)
    thread = H.new_thread(H.auto_title(args.question), STORE)
    ctx = T.Context()
    domain = get_domain(args.domain) or GENERAL
    answer = _answer(args.question, provider=provider, thread_id=thread.id,
                     paint=paint, ctx=ctx, domain=domain)
    print()
    print(paint.dim(f"  thread {thread.id}"))
    return 0 if answer is not None and not answer.error else 1


def _no_provider(name: str, paint: _Paint) -> int:
    if name:
        print(paint.bad(f"No provider matches {name!r}."), file=sys.stderr)
        print("  Refused rather than substituted: an answer attributed to a "
              "model that\n  never saw the question is worse than no answer.",
              file=sys.stderr)
        print("  `alelyon-workspace models` lists what is available.",
              file=sys.stderr)
        return 2
    print(paint.bad("Nothing can answer."), file=sys.stderr)
    print("  Run `alelyon-workspace models` for what Workspace looks for.",
          file=sys.stderr)
    return 1


# ── the conversation ────────────────────────────────────────────────────────

_BANNER = """\
Workspace — Lattice at a terminal.

  The model may reason, calculate and explain. Under every answer, figures that
  came from a deterministic tool are shown apart from figures that are the
  model's own. Nothing here can place, cancel or modify anything.

  /new [title]   start a fresh conversation      /threads   list conversations
  /resume <id>   continue an earlier one         /models    who can answer
  /provider <n>  switch model                    /help      this
  /quit          leave                           Ctrl-C     stop an answer
"""


def _repl(args, paint: _Paint, thread_id: str = "") -> int:
    _store_root()
    provider = _resolve_provider(args.model)
    if provider is None:
        return _no_provider(args.model, paint)

    domain = get_domain(args.domain) or GENERAL
    ctx = T.Context()

    if thread_id:
        thread = next((t for t in H.list_threads(STORE) if t.id == thread_id),
                      None)
        if thread is None:
            print(paint.bad(f"No conversation {thread_id!r}."), file=sys.stderr)
            return 2
        print(_BANNER)
        print(paint.dim(f"  resumed: {thread.title}  ({thread.turns} turns)"))
        for turn in H.load_thread(thread_id, store=STORE)[-6:]:
            who = "you" if turn.role == H.ROLE_USER else "Lattice"
            body = turn.text.strip().splitlines()
            head = body[0][:100] if body else ""
            more = " …" if (len(body) > 1 or len(head) == 100) else ""
            print(paint.dim(f"    {who}: {head}{more}"))
    else:
        thread = H.new_thread("", STORE)
        thread_id = thread.id
        print(_BANNER)

    print(paint.dim(f"  {_describe_provider(provider, paint)}"))
    print(paint.dim(f"  thread {thread_id}"))
    print()

    while True:
        try:
            line = input(paint.bold("you › "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            outcome = _slash(text, paint)
            if outcome is None:
                return 0
            if isinstance(outcome, tuple):
                kind, value = outcome
                if kind == "thread":
                    thread_id = value
                    print(paint.dim(f"  thread {thread_id}"))
                elif kind == "provider":
                    found = _resolve_provider(value)
                    if found is None:
                        print(paint.bad(f"  no provider matches {value!r}; "
                                        f"keeping {provider.name}"))
                    else:
                        provider = found
                        print(paint.dim(f"  {_describe_provider(provider, paint)}"))
            continue

        _answer(text, provider=provider, thread_id=thread_id, paint=paint,
                ctx=ctx, domain=domain)
        print()


def _slash(text: str, paint: _Paint):
    """Handle a slash command.

    Returns None to quit, a `(kind, value)` pair for state the caller owns, or
    anything else to continue.
    """
    command, _, rest = text[1:].partition(" ")
    rest = rest.strip()
    command = command.lower()

    if command in ("quit", "exit", "q"):
        return None
    if command in ("help", "?"):
        print(_BANNER)
        return True
    if command == "models":
        _cmd_models(None, paint)
        return True
    if command == "threads":
        _cmd_threads(None, paint)
        return True
    if command == "new":
        thread = H.new_thread(rest, STORE)
        return ("thread", thread.id)
    if command == "resume":
        if not rest:
            print(paint.bad("  /resume needs a thread id; /threads lists them"))
            return True
        if not any(t.id == rest for t in H.list_threads(STORE)):
            print(paint.bad(f"  no conversation {rest!r}"))
            return True
        return ("thread", rest)
    if command == "provider":
        if not rest:
            print(paint.bad("  /provider needs a name; /models lists them"))
            return True
        return ("provider", rest)
    print(paint.bad(f"  unknown command {text.split()[0]!r}; /help for the list"))
    return True


# ── entry point ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alelyon-workspace",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_BANNER)
    parser.add_argument("--model", default="",
                        help="provider name or prefix; refused rather than "
                             "substituted when it matches nothing")
    parser.add_argument("--domain", default=GENERAL.key,
                        help="which assistant this is. Defaults to the general "
                             "one, whose tools are the certified calculator and "
                             "the model's own declared anatomy")
    sub = parser.add_subparsers(dest="command")

    one = sub.add_parser("ask", help="ask once and exit")
    one.add_argument("question")

    sub.add_parser("threads", help="list conversations")
    sub.add_parser("models", help="what can answer, and from where")

    resume = sub.add_parser("resume", help="continue an earlier conversation")
    resume.add_argument("thread_id")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paint = _Paint(_colour_ok(sys.stdout))

    if args.command == "models":
        return _cmd_models(args, paint)
    if args.command == "threads":
        return _cmd_threads(args, paint)
    if args.command == "ask":
        return _cmd_ask(args, paint)
    if args.command == "resume":
        return _repl(args, paint, thread_id=args.thread_id)
    return _repl(args, paint)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
