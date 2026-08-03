"""Lexer for the Alelyon modeling language (roadmap Phase-5, first artifact).

The vision (owner, restated across sessions): let a researcher express a
quantitative model as plain logical statements, not "syntax mumbo jumbo". The
honest first version is a SMALL, SAFE expression language that is sugar over the
stable object model (DataService series + research transforms) — the roadmap's
own guidance ("a DSL is only ever sugar over a stable object model"). This file
turns source text into tokens; parser.py builds the AST; interpreter.py evaluates
it against real data. No eval/exec anywhere — only a whitelisted vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

KEYWORDS = {"let", "signal", "show", "when", "and", "or", "not"}

# multi-char operators must be tried before single-char
_OPERATORS = [">=", "<=", "==", "!=", ">", "<", "+", "-", "*", "/", "%", "^",
              "(", ")", ",", "="]


class DSLSyntaxError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"line {line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col


@dataclass(frozen=True)
class Token:
    type: str          # NUMBER | STRING | NAME | KEYWORD | OP | NEWLINE | EOF
    value: object
    line: int
    col: int


def tokenize(src: str, *, max_tokens: int | None = None) -> List[Token]:
    """Turn source into a token list terminated by EOF. Raises DSLSyntaxError on
    an unrecognized character or an unterminated string.

    ``max_tokens`` excludes the synthetic EOF token.  The parser supplies a
    finite value at every untrusted DSL boundary so tokenization cannot turn a
    compact source string into an unbounded token vector.
    """
    tokens: List[Token] = []
    i, n = 0, len(src)
    line, line_start = 1, 0

    def col() -> int:
        return i - line_start + 1

    def emit(token: Token) -> None:
        if max_tokens is not None and len(tokens) >= max_tokens:
            raise DSLSyntaxError(
                f"program exceeds DSL token limit of {max_tokens}",
                token.line,
                token.col,
            )
        tokens.append(token)

    while i < n:
        c = src[i]
        # newline → statement separator
        if c == "\n":
            emit(Token("NEWLINE", "\n", line, col()))
            i += 1
            line += 1
            line_start = i
            continue
        # Only the six ASCII whitespace characters are portable across the
        # Python/Rust Profile-1 boundary. Newline is handled above because it is
        # a statement separator; the other five are insignificant spacing.
        if c in " \t\r\v\f":
            i += 1
            continue
        # ';' is an explicit statement separator (same as newline)
        if c == ";":
            emit(Token("NEWLINE", ";", line, col()))
            i += 1
            continue
        # comment to end of line
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        # String literals use JSON's double-quoted grammar. Decoding here keeps
        # escape handling identical to the Rust verifier and rejects ambiguous
        # raw/single-quoted spellings before they enter the AST.
        if c == '"':
            start = i
            start_col = col()
            i += 1
            escaped = False
            while i < n:
                if src[i] == "\n":
                    raise DSLSyntaxError("unterminated string", line, start_col)
                if escaped:
                    escaped = False
                elif src[i] == "\\":
                    escaped = True
                elif src[i] == '"':
                    i += 1
                    break
                i += 1
            else:
                raise DSLSyntaxError("unterminated string", line, start_col)
            raw = src[start:i]
            try:
                value = json.loads(raw)
                # CPython's JSON decoder preserves an unpaired surrogate in a
                # str. CNE source is UTF-8 and Rust String is Unicode-scalar
                # only, so require strict UTF-8 encodability after decoding.
                value.encode("utf-8")
            except (json.JSONDecodeError, UnicodeEncodeError) as exc:
                raise DSLSyntaxError(
                    "invalid JSON string literal", line, start_col) from exc
            emit(Token("STRING", value, line, start_col))
            continue
        # number
        if c.isascii() and (c.isdigit() or (
                c == "." and i + 1 < n
                and src[i + 1].isascii() and src[i + 1].isdigit())):
            start = i
            start_col = col()
            has_dot = False
            while i < n and ((src[i].isascii() and src[i].isdigit())
                             or src[i] == "."
                             or src[i] in "eE"
                             or (src[i] in "+-" and src[i - 1] in "eE")):
                if src[i] == ".":
                    if has_dot:
                        break
                    has_dot = True
                i += 1
            text = src[start:i]
            try:
                val = float(text)
            except ValueError:
                raise DSLSyntaxError(f"bad number {text!r}", line, start_col)
            emit(Token("NUMBER", val, line, start_col))
            continue
        # identifier / keyword
        if c.isascii() and (c.isalpha() or c == "_"):
            start = i
            start_col = col()
            while i < n and src[i].isascii() \
                    and (src[i].isalnum() or src[i] == "_"):
                i += 1
            word = src[start:i]
            ttype = "KEYWORD" if word in KEYWORDS else "NAME"
            emit(Token(ttype, word, line, start_col))
            continue
        # operator
        matched = None
        for op in _OPERATORS:
            if src.startswith(op, i):
                matched = op
                break
        if matched is None:
            raise DSLSyntaxError(f"unexpected character {c!r}", line, col())
        emit(Token("OP", matched, line, col()))
        i += len(matched)

    tokens.append(Token("EOF", None, line, col()))
    return tokens
