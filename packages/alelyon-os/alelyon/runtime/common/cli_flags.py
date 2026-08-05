"""Flags a caller may put on either side of a subcommand.

`argparse` binds an option to the parser that declared it. A flag declared on
the top-level parser is therefore accepted **only before** the subcommand, and
one declared on a subparser only after it. Neither position is more natural, so
whichever a tool picks, half of the invocations people actually type are a usage
error — and argparse reports it as `unrecognized arguments`, which reads like the
flag does not exist rather than like it is in the wrong place.

This was not hypothetical. Measured at `9533bde`, **every documented invocation
of `tools/mail.py` failed**: four in its own module docstring, four in
`README.md`, and two more that the tool *printed at the user* as the command to
run next — all of them writing `--session` after the subcommand, where the flag
was declared before it. `tools/fleet.py` and `tools/relay.py` had the same shape
and were saved only by their docs happening to show the other order.

The trap is that nothing is wrong at the definition site. The flag is declared
once, correctly, and the failure appears at a call site that reads fine and in
documentation that reads fine. Nobody rereads a `--help` they already wrote.

    Declaring a flag once binds it to one position. Accepting it in both
    requires declaring it twice, and the second declaration must not overwrite
    the first with its own default.

That second clause is the part that makes a naive fix worse than the bug. Give
the subparser the same `default=""` and `tool --session X sub` starts *silently*
resolving to `""` — argparse applies the subparser's default after parsing the
top-level value, so a flag that used to error now succeeds with the wrong value.
An error is a bad outcome; silently discarding what the caller typed is a worse
one. `argparse.SUPPRESS` is what avoids it: the trailing copy sets the attribute
only when the caller actually passed it.

Why the subcommand wrapper exists
---------------------------------
The flags have to be attached to **every** subparser, and a subcommand added
later is exactly as broken as the ones this fixed — with no failing test to say
so, because the missing flag is a usage error in a position nobody covers.
`subcommands()` attaches them at the point subparsers are created, so a new
subcommand inherits them by construction rather than by the author remembering.

What this does not do
---------------------
* It does not make the two positions mean different things. `tool --session A
  sub --session B` is last-wins, which is argparse's ordinary behaviour for a
  repeated option and is not special-cased here.
* It does not unify the flags themselves. Each tool still declares its own, so
  two tools can drift apart in what they accept; this makes each tool's own set
  consistent across its subcommands and nothing wider.
"""
from __future__ import annotations

import argparse
from typing import Any, Iterable, Sequence


class _Subcommands:
    """`add_subparsers()` with the shared flags attached to every subparser.

    Delegates everything else, so a caller uses it exactly like the object it
    wraps. Wrapping rather than patching the real one keeps the behaviour
    visible in a traceback instead of appearing as an argparse internal that
    does something argparse does not document.
    """

    def __init__(self, sub, trailing: argparse.ArgumentParser) -> None:
        self._sub = sub
        self._trailing = trailing

    def add_parser(self, name: str, **kwargs) -> argparse.ArgumentParser:
        parents = list(kwargs.pop("parents", ()))
        parents.append(self._trailing)
        return self._sub.add_parser(name, parents=parents, **kwargs)

    def __getattr__(self, item: str) -> Any:      # pragma: no cover - passthrough
        return getattr(self._sub, item)


def either_side(
    specs: Iterable[tuple[Sequence[str], dict]],
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Two parent parsers declaring one set of flags: leading, then trailing.

    `leading` carries the real defaults and belongs on the top-level parser.
    `trailing` declares the same flags with `default=argparse.SUPPRESS` and
    belongs on every subparser, so a caller who omits the flag after the
    subcommand leaves whatever they passed before it untouched.

    Passing `trailing` to the top-level parser instead would typecheck and would
    remove every default, which is why they are returned as an ordered pair
    rather than as one parser used twice.
    """
    leading = argparse.ArgumentParser(add_help=False)
    trailing = argparse.ArgumentParser(add_help=False)
    for flags, options in specs:
        leading.add_argument(*flags, **options)
        # The help text is dropped from the trailing copy on purpose: argparse
        # would otherwise print the same flag in every subcommand's --help, and
        # a flag documented in eleven places is a flag nobody reads once.
        quiet = {key: value for key, value in options.items() if key != "help"}
        quiet["default"] = argparse.SUPPRESS
        quiet["help"] = argparse.SUPPRESS
        trailing.add_argument(*flags, **quiet)
    return leading, trailing


def subcommands(parser: argparse.ArgumentParser,
                trailing: argparse.ArgumentParser, **kwargs) -> _Subcommands:
    """`parser.add_subparsers(**kwargs)`, with `trailing` on every subparser."""
    return _Subcommands(parser.add_subparsers(**kwargs), trailing)
