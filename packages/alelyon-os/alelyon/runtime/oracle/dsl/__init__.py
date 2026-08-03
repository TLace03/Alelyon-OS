"""The Alelyon modeling language (roadmap Phase-5, first artifact).

A small, SAFE declarative language for expressing quant/econ models as plain
logical statements — sugar over the stable object model (DataService + research
transforms). No eval/exec; a fixed whitelisted vocabulary. Example:

    let spy = price("SPY")
    let mom = zscore(returns(spy, 63))
    signal buy when mom > 1 and rsi(spy, 14) < 30
    show corr(returns(spy), returns(price("QQQ")))

Entry point: run_program(src, data_service=...) -> Result.
"""
from alelyon.runtime.oracle.dsl.interpreter import (  # noqa: F401
    BUILTINS,
    DataContext,
    DSLError,
    Interpreter,
    Output,
    Result,
    builtin_names,
    run_program,
)
from alelyon.runtime.oracle.dsl.lexer import DSLSyntaxError, tokenize  # noqa: F401
from alelyon.runtime.oracle.dsl.parser import parse  # noqa: F401

__all__ = [
    "run_program", "Result", "Output", "DataContext", "DSLError", "Interpreter",
    "BUILTINS", "builtin_names", "parse", "tokenize", "DSLSyntaxError",
]
