"""AST node types for the Alelyon modeling language. Plain dataclasses (named
`nodes` to avoid shadowing stdlib `ast`)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

# ── expressions ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Num:
    value: float

@dataclass(frozen=True)
class Str:
    value: str

@dataclass(frozen=True)
class Name:
    id: str

@dataclass(frozen=True)
class Call:
    func: str
    args: list
    line: int = 0

@dataclass(frozen=True)
class BinOp:
    op: str                 # + - * / % ^  or  > < >= <= == !=
    left: object
    right: object

@dataclass(frozen=True)
class BoolOp:
    op: str                 # 'and' | 'or'
    left: object
    right: object

@dataclass(frozen=True)
class Not:
    operand: object

@dataclass(frozen=True)
class UnaryMinus:
    operand: object

Expr = Union[Num, Str, Name, Call, BinOp, BoolOp, Not, UnaryMinus]

# ── statements ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Let:
    name: str
    expr: object
    line: int = 0

@dataclass(frozen=True)
class Signal:
    name: str
    expr: object
    line: int = 0

@dataclass(frozen=True)
class Show:
    expr: object
    src: str = ""
    line: int = 0

Stmt = Union[Let, Signal, Show]

@dataclass(frozen=True)
class Program:
    statements: List[Stmt] = field(default_factory=list)
