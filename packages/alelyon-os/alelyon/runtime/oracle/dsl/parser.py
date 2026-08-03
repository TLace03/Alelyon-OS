"""Recursive-descent parser for the Alelyon modeling language.

Grammar (precedence low→high):
    program    := (statement (NEWLINE)+)*
    statement  := 'let' NAME '=' expr | 'signal' NAME 'when' expr | 'show' expr | expr
    expr       := or_expr
    or_expr    := and_expr ('or' and_expr)*
    and_expr   := not_expr ('and' not_expr)*
    not_expr   := 'not' not_expr | comparison
    comparison := additive (('>'|'<'|'>='|'<='|'=='|'!=') additive)?
    additive   := term (('+'|'-') term)*
    term       := factor (('*'|'/'|'%') factor)*
    factor     := ('-'|'+') factor | power
    power      := atom ('^' factor)?          # right-associative
    atom       := NUMBER | STRING | NAME | call | '(' expr ')'
    call       := NAME '(' (expr (',' expr)*)? ')'
"""
from __future__ import annotations

from typing import List

from alelyon.runtime.oracle.dsl.lexer import DSLSyntaxError, Token, tokenize
from alelyon.runtime.oracle.dsl.nodes import (
    BinOp, BoolOp, Call, Let, Name, Not, Num, Program, Show, Signal, Str,
    UnaryMinus,
)

_CMP = {">", "<", ">=", "<=", "==", "!="}

# Profile-1 verifier resource policy.  Rust mirrors these exact values in
# ``alelyon/languages/cne_verify/src/replay.rs`` and the differential harness
# probes both implementations at the boundaries.
MAX_DSL_SOURCE_BYTES = 64 * 1024
MAX_DSL_TOKENS = 16_384
MAX_DSL_AST_NODES = 8_192
MAX_DSL_AST_DEPTH = 64


class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    # ── cursor ────────────────────────────────────────────────────────────────
    def _peek(self) -> Token:
        return self.toks[self.pos]

    def _advance(self) -> Token:
        t = self.toks[self.pos]
        if t.type != "EOF":
            self.pos += 1
        return t

    def _check(self, ttype: str, value=None) -> bool:
        t = self._peek()
        return t.type == ttype and (value is None or t.value == value)

    def _expect(self, ttype: str, value=None) -> Token:
        t = self._peek()
        if t.type != ttype or (value is not None and t.value != value):
            want = value if value is not None else ttype
            raise DSLSyntaxError(f"expected {want!r}, got {t.value!r}", t.line, t.col)
        return self._advance()

    def _skip_newlines(self) -> None:
        while self._check("NEWLINE"):
            self._advance()

    def _guard_depth(self, depth: int) -> None:
        if depth > MAX_DSL_AST_DEPTH:
            t = self._peek()
            raise DSLSyntaxError(
                f"program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}",
                t.line,
                t.col,
            )

    # ── program / statements ──────────────────────────────────────────────────
    def parse_program(self) -> Program:
        stmts = []
        self._skip_newlines()
        while not self._check("EOF"):
            stmts.append(self._statement())
            if self._check("EOF"):
                break
            # statements must be separated by a newline/';'
            if not self._check("NEWLINE"):
                t = self._peek()
                raise DSLSyntaxError(
                    f"unexpected {t.value!r} after statement (missing newline?)",
                    t.line, t.col)
            self._skip_newlines()
        return Program(stmts)

    def _statement(self):
        t = self._peek()
        if self._check("KEYWORD", "let"):
            self._advance()
            name = self._expect("NAME").value
            self._expect("OP", "=")
            return Let(name, self._expr(1), t.line)
        if self._check("KEYWORD", "signal"):
            self._advance()
            name = self._expect("NAME").value
            self._expect("KEYWORD", "when")
            return Signal(name, self._expr(1), t.line)
        if self._check("KEYWORD", "show"):
            self._advance()
            return Show(self._expr(1), line=t.line)
        # bare expression = show
        return Show(self._expr(1), line=t.line)

    # ── expressions ───────────────────────────────────────────────────────────
    def _expr(self, depth: int):
        self._guard_depth(depth)
        return self._or(depth)

    def _or(self, depth: int):
        left = self._and(depth)
        while self._check("KEYWORD", "or"):
            self._advance()
            left = BoolOp("or", left, self._and(depth))
        return left

    def _and(self, depth: int):
        left = self._not(depth)
        while self._check("KEYWORD", "and"):
            self._advance()
            left = BoolOp("and", left, self._not(depth))
        return left

    def _not(self, depth: int):
        self._guard_depth(depth)
        if self._check("KEYWORD", "not"):
            self._advance()
            return Not(self._not(depth + 1))
        return self._comparison(depth)

    def _comparison(self, depth: int):
        left = self._additive(depth)
        t = self._peek()
        if t.type == "OP" and t.value in _CMP:
            self._advance()
            return BinOp(t.value, left, self._additive(depth))
        return left

    def _additive(self, depth: int):
        left = self._term(depth)
        while self._peek().type == "OP" and self._peek().value in ("+", "-"):
            op = self._advance().value
            left = BinOp(op, left, self._term(depth))
        return left

    def _term(self, depth: int):
        left = self._factor(depth)
        while self._peek().type == "OP" and self._peek().value in ("*", "/", "%"):
            op = self._advance().value
            left = BinOp(op, left, self._factor(depth))
        return left

    def _factor(self, depth: int):
        self._guard_depth(depth)
        t = self._peek()
        if t.type == "OP" and t.value == "-":
            self._advance()
            return UnaryMinus(self._factor(depth + 1))
        if t.type == "OP" and t.value == "+":
            self._advance()
            return self._factor(depth + 1)
        return self._power(depth)

    def _power(self, depth: int):
        base = self._atom(depth)
        if self._check("OP", "^"):
            self._advance()
            return BinOp("^", base, self._factor(depth + 1))   # right-assoc
        return base

    def _atom(self, depth: int):
        self._guard_depth(depth)
        t = self._peek()
        if t.type == "NUMBER":
            self._advance()
            return Num(float(t.value))
        if t.type == "STRING":
            self._advance()
            return Str(str(t.value))
        if t.type == "NAME":
            self._advance()
            if self._check("OP", "("):
                return self._call(t.value, t.line, depth)
            return Name(t.value)
        if t.type == "OP" and t.value == "(":
            self._advance()
            e = self._expr(depth + 1)
            self._expect("OP", ")")
            return e
        raise DSLSyntaxError(f"unexpected {t.value!r}", t.line, t.col)

    def _call(self, name: str, line: int, depth: int):
        self._expect("OP", "(")
        args = []
        if not self._check("OP", ")"):
            args.append(self._expr(depth + 1))
            while self._check("OP", ","):
                self._advance()
                args.append(self._expr(depth + 1))
        self._expect("OP", ")")
        return Call(name, args, line)


def program_shape(program: Program) -> tuple[int, int]:
    """Return ``(node_count, maximum_expression_depth)`` without recursion."""
    nodes = len(program.statements)
    maximum_depth = 0
    stack = [(statement.expr, 1) for statement in program.statements]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if isinstance(node, Call):
            stack.extend((arg, depth + 1) for arg in node.args)
        elif isinstance(node, (BinOp, BoolOp)):
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
        elif isinstance(node, (Not, UnaryMinus)):
            stack.append((node.operand, depth + 1))
    return nodes, maximum_depth


def _validate_program_shape(program: Program) -> None:
    nodes, depth = program_shape(program)
    if nodes > MAX_DSL_AST_NODES:
        raise DSLSyntaxError(
            f"program exceeds DSL AST node limit of {MAX_DSL_AST_NODES}", 1, 1)
    if depth > MAX_DSL_AST_DEPTH:
        raise DSLSyntaxError(
            f"program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}", 1, 1)


def parse(src: str) -> Program:
    """Tokenize + parse `src` into a Program AST. Raises DSLSyntaxError."""
    if len(src.encode("utf-8")) > MAX_DSL_SOURCE_BYTES:
        raise DSLSyntaxError(
            f"program exceeds DSL byte limit of {MAX_DSL_SOURCE_BYTES}", 1, 1)
    program = Parser(tokenize(src, max_tokens=MAX_DSL_TOKENS)).parse_program()
    _validate_program_shape(program)
    return program
