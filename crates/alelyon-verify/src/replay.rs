//! Frozen Profile-1 DSL replay and DRC width reconstruction.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::data::CanonicalInput;
use crate::kernel::ReplayKernel;

const SAFE_EXACT_INT: f64 = 9_007_199_254_740_992.0;
const MAX_DSL_SOURCE_BYTES: usize = 64 * 1024;
const MAX_DSL_TOKENS: usize = 16_384;
const MAX_DSL_AST_NODES: usize = 8_192;
const MAX_DSL_AST_DEPTH: usize = 64;
const MAX_REPLAY_RESAMPLES: i128 = 10_000;
const MAX_REPLAY_WORK_UNITS: u128 = 50_000_000;

#[derive(Debug, Clone)]
pub struct FetchedInput {
    pub data: CanonicalInput,
    pub deltas: Vec<Option<f64>>,
}

#[derive(Debug, Clone)]
pub struct ReplayParams {
    pub seed: Option<u64>,
    pub seed_invalid: bool,
    pub k: i128,
    pub alpha: f64,
    pub strict: bool,
    pub require_tier: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub fatal: bool,
    pub ok: bool,
    pub refused: bool,
    pub reason: Option<String>,
    pub program_class: String,
    pub level: Option<f64>,
    pub level_exact: bool,
    pub width: Option<f64>,
    pub base_value: Option<f64>,
    pub assumptions: Vec<String>,
    pub branch_sites: Vec<serde_json::Value>,
}

impl ReplayResult {
    fn refusal(reason: impl Into<String>, program_class: impl Into<String>) -> Self {
        Self {
            fatal: false,
            ok: false,
            refused: true,
            reason: Some(reason.into()),
            program_class: program_class.into(),
            level: None,
            level_exact: false,
            width: None,
            base_value: None,
            assumptions: Vec::new(),
            branch_sites: Vec::new(),
        }
    }

    fn fatal() -> Self {
        Self {
            fatal: true,
            ok: false,
            refused: false,
            reason: None,
            program_class: "?".to_owned(),
            level: None,
            level_exact: false,
            width: None,
            base_value: None,
            assumptions: Vec::new(),
            branch_sites: Vec::new(),
        }
    }
}

pub(crate) fn non_text_program_refusal() -> ReplayResult {
    // The Python reference passes the signed value to its parser without
    // stringifying it. A non-text JSON value is therefore a normal replay
    // refusal in the unknown ("?") class, not a different valid program.
    ReplayResult::refusal("program does not parse: source is not text", "?")
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Number(f64),
    Text(String),
    Name(String),
    Call(String, Vec<Expr>),
    UnaryMinus(Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
enum Statement {
    Let(String, Expr),
    Signal(String, Expr),
    Show(Expr),
}

#[derive(Debug, Clone, PartialEq)]
struct Program {
    bindings: Vec<(String, Expr)>,
    output: Expr,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Number(f64),
    Text(String),
    Symbol(char),
    Separator,
    End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError(String);

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ReplayError {}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn parse(source: &str) -> Result<Program, ReplayError> {
        if source.len() > MAX_DSL_SOURCE_BYTES {
            return Err(ReplayError(format!(
                "program exceeds DSL byte limit of {MAX_DSL_SOURCE_BYTES}"
            )));
        }
        let tokens = lex(source)?;
        let mut parser = Self {
            tokens,
            position: 0,
        };
        let mut statements = Vec::new();
        parser.skip_separators();
        while !matches!(parser.peek(), Token::End) {
            statements.push(parser.statement()?);
            if matches!(parser.peek(), Token::End) {
                break;
            }
            if !matches!(parser.peek(), Token::Separator) {
                return Err(ReplayError(
                    "statements must be separated by a newline or ';'".into(),
                ));
            }
            parser.skip_separators();
        }
        validate_statement_shape(&statements)?;
        prune_to_final_output(statements)
    }

    fn statement(&mut self) -> Result<Statement, ReplayError> {
        match self.peek() {
            Token::Ident(ref keyword) if keyword == "let" => {
                self.next();
                let Token::Ident(name) = self.next() else {
                    return Err(ReplayError("let binding requires a name".into()));
                };
                if is_reserved_identifier(&name) {
                    return Err(ReplayError(format!(
                        "reserved keyword {name:?} cannot be a binding name"
                    )));
                }
                self.expect_symbol('=')?;
                Ok(Statement::Let(name, self.expression(0, 1)?))
            }
            Token::Ident(ref keyword) if keyword == "signal" => {
                self.next();
                let Token::Ident(name) = self.next() else {
                    return Err(ReplayError("signal binding requires a name".into()));
                };
                if is_reserved_identifier(&name) {
                    return Err(ReplayError(format!(
                        "reserved keyword {name:?} cannot be a binding name"
                    )));
                }
                match self.next() {
                    Token::Ident(keyword) if keyword == "when" => {}
                    token => {
                        return Err(ReplayError(format!(
                            "signal binding requires 'when', got {token:?}"
                        )));
                    }
                }
                Ok(Statement::Signal(name, self.expression(0, 1)?))
            }
            Token::Ident(ref keyword) if keyword == "show" => {
                self.next();
                Ok(Statement::Show(self.expression(0, 1)?))
            }
            _ => Ok(Statement::Show(self.expression(0, 1)?)),
        }
    }

    fn skip_separators(&mut self) {
        while matches!(self.peek(), Token::Separator) {
            self.next();
        }
    }

    fn expression(&mut self, minimum: u8, depth: usize) -> Result<Expr, ReplayError> {
        if depth > MAX_DSL_AST_DEPTH {
            return Err(ReplayError(format!(
                "program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}"
            )));
        }
        let mut left = match self.next() {
            Token::Number(value) => Expr::Number(value),
            Token::Text(value) => Expr::Text(value),
            Token::Symbol('-') => Expr::UnaryMinus(Box::new(self.expression(50, depth + 1)?)),
            Token::Symbol('+') => self.expression(50, depth + 1)?,
            Token::Symbol('(') => {
                let value = self.expression(0, depth + 1)?;
                self.expect_symbol(')')?;
                value
            }
            Token::Ident(name) if is_reserved_identifier(&name) => {
                return Err(ReplayError(format!(
                    "reserved keyword {name:?} cannot be an expression identifier"
                )));
            }
            Token::Ident(name) => {
                if matches!(self.peek(), Token::Symbol('(')) {
                    self.next();
                    let mut arguments = Vec::new();
                    if !matches!(self.peek(), Token::Symbol(')')) {
                        loop {
                            arguments.push(self.expression(0, depth + 1)?);
                            if matches!(self.peek(), Token::Symbol(',')) {
                                self.next();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect_symbol(')')?;
                    Expr::Call(name, arguments)
                } else {
                    Expr::Name(name)
                }
            }
            token => return Err(ReplayError(format!("unexpected token {token:?}"))),
        };
        while let Token::Symbol(symbol @ ('+' | '-' | '*' | '/' | '%' | '^')) = self.peek() {
            let (precedence, right_associative) = match symbol {
                '+' | '-' => (10, false),
                '*' | '/' | '%' => (20, false),
                '^' => (30, true),
                _ => unreachable!(),
            };
            if precedence < minimum {
                break;
            }
            let operator = symbol.to_string();
            self.next();
            let right = self.expression(
                if right_associative {
                    precedence
                } else {
                    precedence + 1
                },
                depth + 1,
            )?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn peek(&self) -> Token {
        self.tokens
            .get(self.position)
            .cloned()
            .unwrap_or(Token::End)
    }

    fn next(&mut self) -> Token {
        let token = self.peek();
        self.position = self.position.saturating_add(1);
        token
    }

    fn expect_symbol(&mut self, expected: char) -> Result<(), ReplayError> {
        match self.next() {
            Token::Symbol(actual) if actual == expected => Ok(()),
            token => Err(ReplayError(format!("expected {expected:?}, got {token:?}"))),
        }
    }
}

fn is_reserved_identifier(name: &str) -> bool {
    matches!(
        name,
        "let" | "signal" | "show" | "when" | "and" | "or" | "not"
    )
}

fn expression_shape(roots: Vec<&Expr>, statement_nodes: usize) -> (usize, usize) {
    let mut nodes = statement_nodes;
    let mut maximum_depth = 0usize;
    let mut stack = roots
        .into_iter()
        .map(|expression| (expression, 1usize))
        .collect::<Vec<_>>();
    while let Some((expression, depth)) = stack.pop() {
        nodes = nodes.saturating_add(1);
        maximum_depth = maximum_depth.max(depth);
        match expression {
            Expr::Call(_, arguments) => {
                stack.extend(arguments.iter().map(|argument| (argument, depth + 1)));
            }
            Expr::UnaryMinus(value) => stack.push((value, depth + 1)),
            Expr::Binary(_, left, right) => {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            }
            Expr::Number(_) | Expr::Text(_) | Expr::Name(_) => {}
        }
    }
    (nodes, maximum_depth)
}

fn validate_statement_shape(statements: &[Statement]) -> Result<(), ReplayError> {
    let roots = statements
        .iter()
        .map(|statement| match statement {
            Statement::Let(_, expression)
            | Statement::Signal(_, expression)
            | Statement::Show(expression) => expression,
        })
        .collect::<Vec<_>>();
    let (nodes, depth) = expression_shape(roots, statements.len());
    if nodes > MAX_DSL_AST_NODES {
        return Err(ReplayError(format!(
            "program exceeds DSL AST node limit of {MAX_DSL_AST_NODES}"
        )));
    }
    if depth > MAX_DSL_AST_DEPTH {
        return Err(ReplayError(format!(
            "program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}"
        )));
    }
    Ok(())
}

fn program_node_count(program: &Program) -> usize {
    let mut roots = program
        .bindings
        .iter()
        .map(|(_, expression)| expression)
        .collect::<Vec<_>>();
    roots.push(&program.output);
    expression_shape(roots, program.bindings.len() + 1).0
}

fn prune_to_final_output(statements: Vec<Statement>) -> Result<Program, ReplayError> {
    let output_index = statements
        .iter()
        .rposition(|statement| matches!(statement, Statement::Show(_) | Statement::Signal(_, _)))
        .ok_or_else(|| ReplayError("program has no output expression".into()))?;
    let output = match &statements[output_index] {
        Statement::Show(expression) | Statement::Signal(_, expression) => expression.clone(),
        Statement::Let(_, _) => {
            return Err(ReplayError("program has no output expression".into()));
        }
    };

    // The reference pruner treats prior signals as both outputs and bindings.
    // The final signal is only the selected output, while every other let/signal
    // participates in last-wins dependency discovery and source-order retention.
    let binding_map = statements
        .iter()
        .enumerate()
        .filter_map(|(index, statement)| {
            if index == output_index {
                return None;
            }
            match statement {
                Statement::Let(name, expression) | Statement::Signal(name, expression) => {
                    Some((name.clone(), expression.clone()))
                }
                Statement::Show(_) => None,
            }
        })
        .collect::<BTreeMap<_, _>>();
    let mut needed = BTreeSet::new();
    let mut pending = names_in(&output);
    while let Some(name) = pending.pop() {
        if needed.insert(name.clone()) {
            if let Some(expression) = binding_map.get(&name) {
                pending.extend(names_in(expression));
            }
        }
    }
    let bindings = statements
        .into_iter()
        .enumerate()
        .filter_map(|(index, statement)| {
            if index == output_index {
                return None;
            }
            match statement {
                Statement::Let(name, expression) | Statement::Signal(name, expression)
                    if needed.contains(&name) =>
                {
                    Some((name, expression))
                }
                _ => None,
            }
        })
        .collect();
    Ok(Program { bindings, output })
}

fn names_in(expression: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    fn walk(expression: &Expr, names: &mut Vec<String>) {
        match expression {
            Expr::Name(name) => names.push(name.clone()),
            Expr::Call(_, arguments) => {
                for argument in arguments {
                    walk(argument, names);
                }
            }
            Expr::UnaryMinus(value) => walk(value, names),
            Expr::Binary(_, left, right) => {
                walk(left, names);
                walk(right, names);
            }
            Expr::Number(_) | Expr::Text(_) => {}
        }
    }
    walk(expression, &mut names);
    names
}

fn push_token(output: &mut Vec<Token>, token: Token) -> Result<(), ReplayError> {
    if output.len() >= MAX_DSL_TOKENS {
        return Err(ReplayError(format!(
            "program exceeds DSL token limit of {MAX_DSL_TOKENS}"
        )));
    }
    output.push(token);
    Ok(())
}

fn lex(source: &str) -> Result<Vec<Token>, ReplayError> {
    let chars = source.char_indices().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut cursor = 0usize;
    while cursor < chars.len() {
        let (byte_index, ch) = chars[cursor];
        if ch == '\n' || ch == ';' {
            push_token(&mut output, Token::Separator)?;
            cursor += 1;
            continue;
        }
        if ch.is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        if ch == '#' {
            while cursor < chars.len() && chars[cursor].1 != '\n' {
                cursor += 1;
            }
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = byte_index;
            cursor += 1;
            while cursor < chars.len()
                && (chars[cursor].1.is_ascii_alphanumeric() || chars[cursor].1 == '_')
            {
                cursor += 1;
            }
            let end = chars.get(cursor).map_or(source.len(), |item| item.0);
            push_token(&mut output, Token::Ident(source[start..end].to_owned()))?;
            continue;
        }
        if ch.is_ascii_digit() || ch == '.' {
            let start = byte_index;
            cursor += 1;
            while cursor < chars.len()
                && matches!(chars[cursor].1, '0'..='9' | '.' | 'e' | 'E' | '+' | '-')
            {
                let prior = chars[cursor - 1].1;
                let current = chars[cursor].1;
                if matches!(current, '+' | '-') && !matches!(prior, 'e' | 'E') {
                    break;
                }
                cursor += 1;
            }
            let end = chars.get(cursor).map_or(source.len(), |item| item.0);
            let value = source[start..end]
                .parse::<f64>()
                .map_err(|_| ReplayError("invalid numeric literal".into()))?;
            if !value.is_finite() {
                return Err(ReplayError("non-finite numeric literal".into()));
            }
            push_token(&mut output, Token::Number(value))?;
            continue;
        }
        if ch == '"' {
            let start = byte_index;
            cursor += 1;
            let mut escaped = false;
            while cursor < chars.len() {
                let current = chars[cursor].1;
                cursor += 1;
                if escaped {
                    escaped = false;
                } else if current == '\\' {
                    escaped = true;
                } else if current == '"' {
                    break;
                }
            }
            let end = chars.get(cursor).map_or(source.len(), |item| item.0);
            let raw = &source[start..end];
            let value = serde_json::from_str::<String>(raw)
                .map_err(|_| ReplayError("invalid string literal".into()))?;
            push_token(&mut output, Token::Text(value))?;
            continue;
        }
        if matches!(
            ch,
            '(' | ')' | ',' | '+' | '-' | '*' | '/' | '%' | '^' | '='
        ) {
            push_token(&mut output, Token::Symbol(ch))?;
            cursor += 1;
            continue;
        }
        return Err(ReplayError(format!("unsupported character {ch:?}")));
    }
    output.push(Token::End);
    Ok(output)
}

fn inspect_program(program: &Program) -> (Vec<(String, String)>, Vec<String>, Vec<String>) {
    let mut references = Vec::new();
    let mut hard = Vec::new();
    let mut smooth = Vec::new();
    fn walk(
        expression: &Expr,
        references: &mut Vec<(String, String)>,
        hard: &mut Vec<String>,
        smooth: &mut Vec<String>,
    ) {
        match expression {
            Expr::Call(name, arguments) => {
                if matches!(name.as_str(), "price" | "series" | "table") {
                    if let Some(Expr::Text(key)) = arguments.first() {
                        references.push((name.clone(), key.clone()));
                    } else {
                        hard.push(name.clone());
                    }
                } else if name == "rsi" {
                    hard.push(name.clone());
                } else if matches!(
                    name.as_str(),
                    "returns"
                        | "logret"
                        | "sqrt"
                        | "zscore"
                        | "std"
                        | "rolling_std"
                        | "corr"
                        | "abs"
                        | "clip"
                ) {
                    smooth.push(name.clone());
                } else if !matches!(
                    name.as_str(),
                    "diff"
                        | "lag"
                        | "sma"
                        | "ema"
                        | "rolling_mean"
                        | "sum"
                        | "mean"
                        | "last"
                        | "first"
                        | "count"
                ) {
                    hard.push(name.clone());
                }
                for argument in arguments {
                    walk(argument, references, hard, smooth);
                }
            }
            Expr::Binary(operator, left, right) => {
                if matches!(operator.as_str(), "*" | "/" | "^") {
                    smooth.push(operator.clone());
                } else if operator == "%" {
                    hard.push(operator.clone());
                }
                walk(left, references, hard, smooth);
                walk(right, references, hard, smooth);
            }
            Expr::UnaryMinus(value) => walk(value, references, hard, smooth),
            Expr::Number(_) | Expr::Text(_) | Expr::Name(_) => {}
        }
    }
    for (_, expression) in &program.bindings {
        walk(expression, &mut references, &mut hard, &mut smooth);
    }
    walk(&program.output, &mut references, &mut hard, &mut smooth);
    let mut seen = BTreeSet::new();
    references.retain(|value| seen.insert(value.clone()));
    hard.sort();
    hard.dedup();
    smooth.sort();
    smooth.dedup();
    (references, hard, smooth)
}

#[derive(Debug, Clone, PartialEq)]
enum SeriesIndex {
    Time(Vec<f64>),
    Keys(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
struct IndexedSeries {
    index: SeriesIndex,
    values: Vec<f64>,
}

impl IndexedSeries {
    fn from_input(input: &CanonicalInput) -> Self {
        match input {
            CanonicalInput::TimeSeries { index, values } => Self {
                index: SeriesIndex::Time(index.clone()),
                values: values.clone(),
            },
            CanonicalInput::KeyedTable { keys, values } => Self {
                index: SeriesIndex::Keys(keys.clone()),
                values: values.clone(),
            },
        }
    }

    fn with_values(&self, values: Vec<f64>) -> Result<Self, ReplayError> {
        if values.len() != self.values.len() {
            return Err(ReplayError("series value/index length mismatch".into()));
        }
        Ok(Self {
            index: self.index.clone(),
            values,
        })
    }

    fn map_values(self, mut operation: impl FnMut(f64) -> f64) -> Self {
        Self {
            index: self.index,
            values: self.values.into_iter().map(&mut operation).collect(),
        }
    }
}

#[derive(Debug, Clone)]
enum EvalValue {
    Scalar(f64),
    Text(String),
    Series(IndexedSeries),
}

impl EvalValue {
    fn scalar(self) -> Result<f64, ReplayError> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::Series(series) => series
                .values
                .into_iter()
                .rev()
                .find(|value| !value.is_nan())
                .ok_or_else(|| ReplayError("series has no finite scalar output".into())),
            Self::Text(_) => Err(ReplayError("text is not a scalar".into())),
        }
    }
}

fn evaluate_program(
    program: &Program,
    inputs: &BTreeMap<(String, String), IndexedSeries>,
    kernel: &dyn ReplayKernel,
) -> Result<EvalValue, ReplayError> {
    let mut bindings = BTreeMap::new();
    for (name, expression) in &program.bindings {
        let value = evaluate(expression, inputs, &bindings, kernel)?;
        bindings.insert(name.clone(), value);
    }
    evaluate(&program.output, inputs, &bindings, kernel)
}

fn evaluate(
    expression: &Expr,
    inputs: &BTreeMap<(String, String), IndexedSeries>,
    bindings: &BTreeMap<String, EvalValue>,
    kernel: &dyn ReplayKernel,
) -> Result<EvalValue, ReplayError> {
    match expression {
        Expr::Number(value) => Ok(EvalValue::Scalar(*value)),
        Expr::Text(value) => Ok(EvalValue::Text(value.clone())),
        Expr::Name(name) => bindings
            .get(name)
            .cloned()
            .ok_or_else(|| ReplayError(format!("unknown name {name:?}"))),
        Expr::UnaryMinus(value) => unary_minus(evaluate(value, inputs, bindings, kernel)?),
        Expr::Binary(operator, left, right) => binary(
            operator,
            evaluate(left, inputs, bindings, kernel)?,
            evaluate(right, inputs, bindings, kernel)?,
        ),
        Expr::Call(name, arguments) => {
            let values = arguments
                .iter()
                .map(|argument| evaluate(argument, inputs, bindings, kernel))
                .collect::<Result<Vec<_>, _>>()?;
            call(name, values, inputs, kernel)
        }
    }
}

fn call(
    name: &str,
    mut arguments: Vec<EvalValue>,
    inputs: &BTreeMap<(String, String), IndexedSeries>,
    kernel: &dyn ReplayKernel,
) -> Result<EvalValue, ReplayError> {
    if matches!(name, "price" | "series" | "table") {
        if arguments.len() != 1 {
            return Err(ReplayError(format!("{name} expects one argument")));
        }
        let EvalValue::Text(key) = arguments.remove(0) else {
            return Err(ReplayError(format!("{name} key must be text")));
        };
        return inputs
            .get(&(name.to_owned(), key.clone()))
            .cloned()
            .map(EvalValue::Series)
            .ok_or_else(|| ReplayError(format!("data fetch failed for ({name:?}, {key:?})")));
    }
    match (name, arguments.as_slice()) {
        ("sum", [EvalValue::Series(series)]) => {
            Ok(EvalValue::Scalar(kernel.sum(&drop_nan(&series.values))))
        }
        ("mean", [EvalValue::Series(series)]) => {
            Ok(EvalValue::Scalar(kernel.mean(&drop_nan(&series.values))))
        }
        ("count", [EvalValue::Series(series)]) => Ok(EvalValue::Scalar(
            series.values.iter().filter(|value| !value.is_nan()).count() as f64,
        )),
        ("first", [EvalValue::Series(series)]) => Ok(EvalValue::Scalar(
            series
                .values
                .iter()
                .copied()
                .find(|value| !value.is_nan())
                .unwrap_or(f64::NAN),
        )),
        ("last", [EvalValue::Series(series)]) => Ok(EvalValue::Scalar(
            series
                .values
                .iter()
                .rev()
                .copied()
                .find(|value| !value.is_nan())
                .unwrap_or(f64::NAN),
        )),
        ("diff", [EvalValue::Series(series)]) => Ok(EvalValue::Series(
            series.with_values(diff(&series.values, 1))?,
        )),
        ("diff", [EvalValue::Series(series), EvalValue::Scalar(n)]) => Ok(EvalValue::Series(
            series.with_values(diff(&series.values, integer_argument(*n)?))?,
        )),
        ("lag", [EvalValue::Series(series)]) => Ok(EvalValue::Series(
            series.with_values(lag(&series.values, 1))?,
        )),
        ("lag", [EvalValue::Series(series), EvalValue::Scalar(n)]) => Ok(EvalValue::Series(
            series.with_values(lag(&series.values, integer_argument(*n)?))?,
        )),
        ("sma" | "rolling_mean", [EvalValue::Series(series), EvalValue::Scalar(window)]) => {
            Ok(EvalValue::Series(series.with_values(rolling_mean(
                &series.values,
                integer_argument(*window)?,
                kernel,
            ))?))
        }
        ("ema", [EvalValue::Series(series), EvalValue::Scalar(window)]) => Ok(EvalValue::Series(
            series.with_values(ema(&series.values, integer_argument(*window)?.max(1)))?,
        )),
        ("rsi", [EvalValue::Series(series)]) => Ok(EvalValue::Series(
            series.with_values(rsi(&series.values, 14))?,
        )),
        ("rsi", [EvalValue::Series(series), EvalValue::Scalar(window)]) => Ok(EvalValue::Series(
            series.with_values(rsi(&series.values, integer_argument(*window)?))?,
        )),
        _ => Err(ReplayError(format!("unsupported call {name:?}"))),
    }
}

fn integer_argument(value: f64) -> Result<usize, ReplayError> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64 {
        Ok(value as usize)
    } else {
        Err(ReplayError(
            "window/lag argument is not a non-negative integer".into(),
        ))
    }
}

fn drop_nan(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect()
}

fn unary_minus(value: EvalValue) -> Result<EvalValue, ReplayError> {
    match value {
        EvalValue::Scalar(value) => Ok(EvalValue::Scalar(-value)),
        EvalValue::Series(series) => Ok(EvalValue::Series(series.map_values(|value| -value))),
        EvalValue::Text(_) => Err(ReplayError("cannot negate text".into())),
    }
}

fn binary(operator: &str, left: EvalValue, right: EvalValue) -> Result<EvalValue, ReplayError> {
    let apply = |left: f64, right: f64| match operator {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        "/" => left / right,
        "%" => left % right,
        "^" => left.powf(right),
        _ => f64::NAN,
    };
    match (left, right) {
        (EvalValue::Scalar(left), EvalValue::Scalar(right)) => {
            Ok(EvalValue::Scalar(apply(left, right)))
        }
        (EvalValue::Series(left), EvalValue::Scalar(right)) => Ok(EvalValue::Series(
            left.map_values(|left| apply(left, right)),
        )),
        (EvalValue::Scalar(left), EvalValue::Series(right)) => Ok(EvalValue::Series(
            right.map_values(|right| apply(left, right)),
        )),
        (EvalValue::Series(left), EvalValue::Series(right)) => {
            Ok(EvalValue::Series(align_series(left, right, apply)?))
        }
        _ => Err(ReplayError(
            "binary operands have incompatible types or indices".into(),
        )),
    }
}

fn align_series(
    left: IndexedSeries,
    right: IndexedSeries,
    operation: impl Fn(f64, f64) -> f64,
) -> Result<IndexedSeries, ReplayError> {
    if left.values.len()
        != match &left.index {
            SeriesIndex::Time(index) => index.len(),
            SeriesIndex::Keys(index) => index.len(),
        }
        || right.values.len()
            != match &right.index {
                SeriesIndex::Time(index) => index.len(),
                SeriesIndex::Keys(index) => index.len(),
            }
    {
        return Err(ReplayError("series value/index length mismatch".into()));
    }

    match (left.index, right.index) {
        (SeriesIndex::Time(left_index), SeriesIndex::Time(right_index)) => {
            let mut index = Vec::with_capacity(left_index.len().saturating_add(right_index.len()));
            let mut values = Vec::with_capacity(index.capacity());
            let (mut left_position, mut right_position) = (0usize, 0usize);
            while left_position < left_index.len() || right_position < right_index.len() {
                match (
                    left_index.get(left_position),
                    right_index.get(right_position),
                ) {
                    (Some(&left_key), Some(&right_key)) if left_key == right_key => {
                        index.push(left_key);
                        values.push(operation(
                            left.values[left_position],
                            right.values[right_position],
                        ));
                        left_position += 1;
                        right_position += 1;
                    }
                    (Some(&left_key), Some(&right_key))
                        if left_key.total_cmp(&right_key).is_lt() =>
                    {
                        index.push(left_key);
                        values.push(f64::NAN);
                        left_position += 1;
                    }
                    (Some(_), Some(&right_key)) => {
                        index.push(right_key);
                        values.push(f64::NAN);
                        right_position += 1;
                    }
                    (Some(&left_key), None) => {
                        index.push(left_key);
                        values.push(f64::NAN);
                        left_position += 1;
                    }
                    (None, Some(&right_key)) => {
                        index.push(right_key);
                        values.push(f64::NAN);
                        right_position += 1;
                    }
                    (None, None) => break,
                }
            }
            Ok(IndexedSeries {
                index: SeriesIndex::Time(index),
                values,
            })
        }
        (SeriesIndex::Keys(left_index), SeriesIndex::Keys(right_index)) => {
            let mut index = Vec::with_capacity(left_index.len().saturating_add(right_index.len()));
            let mut values = Vec::with_capacity(index.capacity());
            let (mut left_position, mut right_position) = (0usize, 0usize);
            while left_position < left_index.len() || right_position < right_index.len() {
                match (
                    left_index.get(left_position),
                    right_index.get(right_position),
                ) {
                    (Some(left_key), Some(right_key)) if left_key == right_key => {
                        index.push(left_key.clone());
                        values.push(operation(
                            left.values[left_position],
                            right.values[right_position],
                        ));
                        left_position += 1;
                        right_position += 1;
                    }
                    (Some(left_key), Some(right_key)) if left_key < right_key => {
                        index.push(left_key.clone());
                        values.push(f64::NAN);
                        left_position += 1;
                    }
                    (Some(_), Some(right_key)) => {
                        index.push(right_key.clone());
                        values.push(f64::NAN);
                        right_position += 1;
                    }
                    (Some(left_key), None) => {
                        index.push(left_key.clone());
                        values.push(f64::NAN);
                        left_position += 1;
                    }
                    (None, Some(right_key)) => {
                        index.push(right_key.clone());
                        values.push(f64::NAN);
                        right_position += 1;
                    }
                    (None, None) => break,
                }
            }
            Ok(IndexedSeries {
                index: SeriesIndex::Keys(index),
                values,
            })
        }
        _ => Err(ReplayError(
            "binary operands have incompatible types or indices".into(),
        )),
    }
}

fn lag(values: &[f64], n: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| index.checked_sub(n).map_or(f64::NAN, |prior| values[prior]))
        .collect()
}

fn diff(values: &[f64], n: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            index
                .checked_sub(n)
                .map_or(f64::NAN, |prior| values[index] - values[prior])
        })
        .collect()
}

fn rolling_mean(values: &[f64], window: usize, kernel: &dyn ReplayKernel) -> Vec<f64> {
    if window == 0 {
        return vec![f64::NAN; values.len()];
    }
    (0..values.len())
        .map(|index| {
            if index + 1 < window {
                return f64::NAN;
            }
            let slice = &values[index + 1 - window..=index];
            if slice.iter().any(|value| value.is_nan()) {
                f64::NAN
            } else {
                kernel.sum(slice) / window as f64
            }
        })
        .collect()
}

fn ema(values: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let old_weight_factor = 1.0 - alpha;
    let mut weighted = f64::NAN;
    let mut old_weight = 1.0;
    let mut output = Vec::with_capacity(values.len());
    for &current in values {
        let observed = !current.is_nan();
        if weighted.is_nan() {
            if observed {
                weighted = current;
                old_weight = 1.0;
            }
        } else {
            // pandas ewm(..., adjust=false) defaults to ignore_na=false. Missing
            // positions therefore decay the retained observation's weight even
            // though the visible output is carried forward.
            old_weight *= old_weight_factor;
            if observed {
                if weighted != current {
                    weighted = (old_weight * weighted + alpha * current) / (old_weight + alpha);
                }
                old_weight = 1.0;
            }
        }
        output.push(weighted);
    }
    output
}

fn rsi(values: &[f64], window: usize) -> Vec<f64> {
    let differences = diff(values, 1);
    let gains = differences
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let losses = differences
        .iter()
        .map(|value| (-value).max(0.0))
        .collect::<Vec<_>>();
    let alpha = 1.0 / window.max(1) as f64;
    fn ewm(values: &[f64], alpha: f64) -> Vec<f64> {
        let mut prior = None;
        values
            .iter()
            .map(|value| {
                if value.is_nan() {
                    return prior.unwrap_or(f64::NAN);
                }
                let next = prior.map_or(*value, |old| (1.0 - alpha) * old + alpha * value);
                prior = Some(next);
                next
            })
            .collect()
    }
    ewm(&gains, alpha)
        .into_iter()
        .zip(ewm(&losses, alpha))
        .map(|(gain, loss)| {
            if loss == 0.0 {
                f64::NAN
            } else {
                let ratio = gain / loss;
                100.0 - 100.0 / (1.0 + ratio)
            }
        })
        .collect()
}

pub fn replay(
    source: &str,
    fetched: &BTreeMap<(String, String), FetchedInput>,
    params: &ReplayParams,
    kernel: &dyn ReplayKernel,
) -> ReplayResult {
    if !(0.0 < params.alpha && params.alpha < 1.0) {
        return ReplayResult::refusal(format!("alpha must be in (0,1), got {}", params.alpha), "?");
    }
    if source.trim().is_empty() {
        return ReplayResult::refusal("program reads no certified data", "linear-exact");
    }
    let program = match Parser::parse(source) {
        Ok(program) => program,
        Err(error) => {
            return ReplayResult::refusal(format!("program does not parse: {error}"), "?");
        }
    };
    let ast_nodes = program_node_count(&program);
    let (references, hard, smooth) = inspect_program(&program);
    let class = if !hard.is_empty() {
        "branch-sensitive"
    } else if !smooth.is_empty() {
        "smooth-first-order"
    } else {
        "linear-exact"
    };
    let floors = [
        ("linear-exact", 0),
        ("branch-stable-exact", 1),
        ("smooth-first-order", 2),
        ("branch-stable-first-order", 3),
        ("branch-sensitive", 4),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    if let Some(required) = params.require_tier.as_deref() {
        let Some(required_floor) = floors.get(required) else {
            let names = [
                "branch-sensitive",
                "branch-stable-exact",
                "branch-stable-first-order",
                "linear-exact",
                "smooth-first-order",
            ];
            return ReplayResult::refusal(
                format!("require_tier {required:?} is not one of {names:?}"),
                "?",
            );
        };
        if floors[class] > *required_floor {
            let mut offenders = hard.iter().chain(&smooth).cloned().collect::<Vec<_>>();
            offenders.sort();
            offenders.dedup();
            return ReplayResult::refusal(
                format!(
                    "program classifies '{class}' but this issuer requires '{required}': {}. Outside '{required}' the conformal level is no longer exact, so the result would carry a weaker claim than the one asked for",
                    if offenders.is_empty() {
                        "unknown op".to_owned()
                    } else {
                        offenders.join(", ")
                    }
                ),
                class,
            );
        }
    }
    if !hard.is_empty() && params.strict {
        return ReplayResult::refusal(
            format!(
                "branch-sensitive ops — a dither-sized perturbation can flip a discrete branch, no sound bound exists: {}",
                hard.join(", ")
            ),
            class,
        );
    }
    if references.is_empty() {
        return ReplayResult::refusal("program reads no certified data", class);
    }
    let mut n_rows = 0usize;
    let mut n_uncertified = 0usize;
    for reference in &references {
        let Some(input) = fetched.get(reference) else {
            return ReplayResult::refusal(
                format!("data fetch failed: input {reference:?} is absent"),
                class,
            );
        };
        if input.data.len() != input.deltas.len() && input.deltas.len() != 1 {
            return ReplayResult::fatal();
        }
        n_rows += input.data.len();
        n_uncertified += input.deltas.iter().filter(|value| value.is_none()).count();
    }
    if params.k > MAX_REPLAY_RESAMPLES {
        return ReplayResult::refusal(
            format!(
                "K={} exceeds the verifier limit of {MAX_REPLAY_RESAMPLES} resamples",
                params.k
            ),
            class,
        );
    }
    if let Ok(runs) = u128::try_from(params.k.saturating_add(1)) {
        let replay_work = runs
            .saturating_mul(n_rows as u128)
            .saturating_mul(ast_nodes as u128);
        if replay_work > MAX_REPLAY_WORK_UNITS {
            return ReplayResult::refusal(
                format!(
                    "replay work {replay_work} exceeds the verifier limit of {MAX_REPLAY_WORK_UNITS} units (rows={n_rows}, K={}, AST nodes={ast_nodes})",
                    params.k
                ),
                class,
            );
        }
    }
    if params.strict && n_uncertified > 0 {
        return ReplayResult::refusal(
            format!(
                "{n_uncertified}/{n_rows} consumed rows carry no capture certificate (no exact current cert-log membership) — a bound conditional on 'uncertified rows are exact' would be theater"
            ),
            class,
        );
    }
    let mut effective = BTreeMap::new();
    for reference in &references {
        let input = &fetched[reference];
        let mut deltas = if input.deltas.len() == 1 && input.data.len() != 1 {
            vec![input.deltas[0]; input.data.len()]
        } else {
            input.deltas.clone()
        };
        if deltas.iter().any(Option::is_none) {
            let max = deltas.iter().flatten().copied().reduce(f64::max);
            let Some(max) = max else {
                return ReplayResult::refusal(
                    format!(
                        "input {reference:?} is fully uncertified — no capture Δ to bound its storage error against"
                    ),
                    class,
                );
            };
            for delta in &mut deltas {
                if delta.is_none() {
                    *delta = Some(max);
                }
            }
        }
        effective.insert(
            reference.clone(),
            deltas.into_iter().flatten().collect::<Vec<_>>(),
        );
    }
    let base_inputs = references
        .iter()
        .filter_map(|reference| {
            fetched
                .get(reference)
                .map(|input| (reference.clone(), IndexedSeries::from_input(&input.data)))
        })
        .collect::<BTreeMap<_, _>>();
    let base_value = match evaluate_program(&program, &base_inputs, kernel)
        .and_then(EvalValue::scalar)
    {
        Ok(value) if value.is_finite() => value,
        Ok(_) => return ReplayResult::refusal("base execution failed: non-finite output", class),
        Err(error) => {
            return ReplayResult::refusal(format!("base execution failed: {error}"), class);
        }
    };
    let exact_storage = n_rows > 0
        && effective
            .values()
            .all(|deltas| deltas.iter().all(|delta| *delta == 0.0));
    let integral_inputs = exact_storage
        && base_inputs.values().all(|series| {
            let finite = series
                .values
                .iter()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            !finite.is_empty()
                && finite
                    .iter()
                    .all(|value| value.abs() <= SAFE_EXACT_INT && value.fract() == 0.0)
        });
    if integral_inputs && base_value.abs() > SAFE_EXACT_INT {
        return ReplayResult::refusal(
            format!(
                "a result of {base_value:?} over exactly-stored integer inputs exceeds 2^53 (~$90 trillion in cents): beyond this f64 no longer represents consecutive integers, so the value cannot be claimed exact even though every input element was — the guard binds the aggregate, not only the elements"
            ),
            class,
        );
    }
    let m = ((1.0 - params.alpha) * (params.k + 1) as f64).ceil() as i128;
    if m > params.k {
        return ReplayResult::refusal(
            format!("K={} too small for alpha={}", params.k, params.alpha),
            class,
        );
    }
    let Ok(k_count) = usize::try_from(params.k) else {
        return ReplayResult::refusal(
            format!("K={} too small for alpha={}", params.k, params.alpha),
            class,
        );
    };
    if params.seed_invalid {
        return ReplayResult::fatal();
    }
    let seed = params.seed.unwrap_or(0);
    let mut deviations = Vec::with_capacity(k_count);
    for k in 0..k_count {
        let mut generator = kernel.dither(seed, k as u64);
        let mut perturbed = BTreeMap::new();
        for reference in &references {
            let series = &base_inputs[reference];
            let deltas = &effective[reference];
            let offsets = match generator.resample(deltas) {
                Ok(offsets) => offsets,
                Err(error) => {
                    return ReplayResult::refusal(
                        format!(
                            "perturbed run {}/{} failed ({error}) — the program is unstable at dither scale",
                            k + 1,
                            params.k
                        ),
                        class,
                    );
                }
            };
            perturbed.insert(
                reference.clone(),
                match series.with_values(
                    series
                        .values
                        .iter()
                        .zip(offsets)
                        .map(|(value, offset)| value + offset)
                        .collect(),
                ) {
                    Ok(series) => series,
                    Err(_) => return ReplayResult::fatal(),
                },
            );
        }
        let value = match evaluate_program(&program, &perturbed, kernel).and_then(EvalValue::scalar)
        {
            Ok(value) if value.is_finite() => value,
            _ => {
                return ReplayResult::refusal(
                    format!(
                        "perturbed run {}/{} failed (non-finite output) — the program is unstable at dither scale",
                        k + 1,
                        params.k
                    ),
                    class,
                );
            }
        };
        deviations.push((value - base_value).abs());
    }
    deviations.sort_by(f64::total_cmp);
    let width = deviations[usize::try_from(m - 1).expect("positive order statistic")];
    let level = m as f64 / (params.k + 1) as f64;
    let mut assumptions = vec![
        "resampling PRNG independent of capture PRNG (fresh seed, persisted)".to_owned(),
        "covers STORAGE QUANTIZATION only — sampling/provider/model error are separate terms"
            .to_owned(),
    ];
    if exact_storage {
        assumptions.push(
            "every consumed input was stored EXACTLY (Δ = 0 on every row — e.g. the exact-cents monetary capture law), so the storage-quantization term is genuinely zero rather than merely small. This bounds STORAGE only: floating-point rounding in the computation itself (the division in a mean, say) is a compute-side error this certificate does not cover".to_owned()
                + if integral_inputs { ", and the aggregate was checked to lie within f64's exact-integer range" } else { "" },
        );
    }
    if !smooth.is_empty() {
        assumptions.push(
            "exchangeability holds to first order for smooth ops at capture deltas; level is approximate and harness-validated, not a theorem".to_owned(),
        );
    }
    ReplayResult {
        fatal: false,
        ok: true,
        refused: false,
        reason: None,
        program_class: class.to_owned(),
        level: Some(level),
        level_exact: n_uncertified == 0 && class == "linear-exact",
        width: Some(width),
        base_value: Some(base_value),
        assumptions,
        branch_sites: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::CanonicalInput;
    use crate::kernel::TestKernel;

    fn test_params() -> ReplayParams {
        ReplayParams {
            seed: Some(7),
            seed_invalid: false,
            k: 3,
            alpha: 0.25,
            strict: true,
            require_tier: None,
        }
    }

    fn time_input(index: Vec<f64>, values: Vec<f64>) -> FetchedInput {
        let deltas = vec![Some(0.0); values.len()];
        FetchedInput {
            data: CanonicalInput::TimeSeries { index, values },
            deltas,
        }
    }

    fn table_input(keys: &[&str], values: Vec<f64>) -> FetchedInput {
        let deltas = vec![Some(0.0); values.len()];
        FetchedInput {
            data: CanonicalInput::KeyedTable {
                keys: keys.iter().map(|key| (*key).to_owned()).collect(),
                values,
            },
            deltas,
        }
    }

    #[test]
    fn parses_and_replays_the_golden_mean_profile() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "SYN".into()),
            FetchedInput {
                data: CanonicalInput::TimeSeries {
                    index: vec![1.0, 2.0],
                    values: vec![1.0, 3.0],
                },
                deltas: vec![Some(0.0), Some(0.0)],
            },
        );
        let result = replay(
            "show mean(price(\"SYN\"))",
            &fetched,
            &ReplayParams {
                seed: Some(7),
                seed_invalid: false,
                k: 63,
                alpha: 0.05,
                strict: true,
                require_tier: None,
            },
            &TestKernel,
        );
        assert!(result.ok);
        assert_eq!(result.base_value, Some(2.0));
        assert_eq!(result.width, Some(0.0));
        assert_eq!(result.level, Some(0.953125));
    }

    #[test]
    fn portable_profile1_grammar_accepts_bare_unary_and_json_strings() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "SYN".into()),
            time_input(vec![1.0, 2.0], vec![1.0, 3.0]),
        );
        for source in [
            r#"mean(price("SYN"))"#,
            r#"show +mean(price("SYN"))"#,
            r#"show mean(price("\u0053\u0059\u004e"))"#,
        ] {
            let result = replay(source, &fetched, &test_params(), &TestKernel);
            assert!(result.ok, "{source:?}: {result:?}");
            assert_eq!(result.base_value, Some(2.0));
        }

        Parser::parse(r#"show "\ud835\udc0c""#)
            .expect("a valid JSON surrogate pair must decode to one Unicode scalar");
        for source in [
            r#"show mean(price('SYN'))"#,
            r#"let π = price("SYN"); show mean(π)"#,
            r#"let show = price("SYN"); show mean(show)"#,
            "show ١",
            "show\u{00a0}mean(price(\"SYN\"))",
            r#"show "\ud800""#,
        ] {
            Parser::parse(source).expect_err("non-portable grammar must be refused");
        }
    }

    #[test]
    fn signals_are_outputs_bindings_and_pruned_when_unused() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "SYN".into()),
            time_input(vec![1.0, 2.0], vec![1.0, 3.0]),
        );
        for source in [
            concat!(
                "signal ignored when mean(price(\"MISSING\")); ",
                "show mean(price(\"SYN\"))"
            ),
            "signal score when mean(price(\"SYN\"))",
            "signal score when mean(price(\"SYN\")); show score",
        ] {
            let result = replay(source, &fetched, &test_params(), &TestKernel);
            assert!(result.ok, "{source:?}: {result:?}");
            assert_eq!(result.base_value, Some(2.0));
        }
    }

    #[test]
    fn aggregate_replay_work_is_bounded_before_resampling() {
        let term = "mean(price(\"SYN\"))";
        let source = format!("show {}", vec![term; 15].join(" + "));
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "SYN".into()),
            time_input(
                (0..120).map(f64::from).collect(),
                (0..120).map(f64::from).collect(),
            ),
        );
        let result = replay(
            &source,
            &fetched,
            &ReplayParams {
                seed: Some(123),
                seed_invalid: false,
                k: 10_000,
                alpha: 0.05,
                strict: true,
                require_tier: None,
            },
            &TestKernel,
        );
        assert!(result.refused);
        assert_eq!(
            result.reason.as_deref(),
            Some(
                "replay work 72007200 exceeds the verifier limit of 50000000 units (rows=120, K=10000, AST nodes=60)"
            )
        );
    }

    #[test]
    fn untrusted_dsl_resource_limits_refuse_without_recursing_unboundedly() {
        let cases = [
            (
                "x".repeat(MAX_DSL_SOURCE_BYTES + 1),
                format!("program exceeds DSL byte limit of {MAX_DSL_SOURCE_BYTES}"),
            ),
            (
                vec!["show 0"; 5_462].join(";"),
                format!("program exceeds DSL token limit of {MAX_DSL_TOKENS}"),
            ),
            (
                vec!["show 0"; 4_097].join(";"),
                format!("program exceeds DSL AST node limit of {MAX_DSL_AST_NODES}"),
            ),
            (
                format!(
                    "show {}0{}",
                    "(".repeat(MAX_DSL_AST_DEPTH),
                    ")".repeat(MAX_DSL_AST_DEPTH)
                ),
                format!("program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}"),
            ),
            (
                format!("show {}", vec!["1"; MAX_DSL_AST_DEPTH + 1].join("+")),
                format!("program exceeds DSL AST depth limit of {MAX_DSL_AST_DEPTH}"),
            ),
        ];
        for (source, expected) in cases {
            let error = Parser::parse(&source).expect_err("resource excess must refuse");
            assert!(
                error.to_string().contains(&expected),
                "expected {expected:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn branch_sensitive_program_refuses_with_protocol_visible_reason() {
        let result = replay(
            "show mean(rsi(price(\"SYN\"), 14))",
            &BTreeMap::new(),
            &ReplayParams {
                seed: None,
                seed_invalid: false,
                k: 63,
                alpha: 0.05,
                strict: true,
                require_tier: None,
            },
            &TestKernel,
        );
        assert!(result.refused);
        assert_eq!(
            result.reason.as_deref(),
            Some(
                "branch-sensitive ops — a dither-sized perturbation can flip a discrete branch, no sound bound exists: rsi"
            )
        );
    }

    #[test]
    fn lets_and_final_output_pruning_ignore_unreachable_failures() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "SYN".into()),
            time_input(vec![1.0, 2.0], vec![1.0, 3.0]),
        );
        let source = concat!(
            "let discarded = mean(rsi(price(\"MISSING\"), 14));\n",
            "show mean(price(\"ALSO_MISSING\"));\n",
            "let raw = price(\"SYN\"); let answer = mean(raw);\n",
            "show answer",
        );
        let result = replay(source, &fetched, &test_params(), &TestKernel);
        assert!(result.ok, "{result:?}");
        assert_eq!(result.program_class, "linear-exact");
        assert_eq!(result.base_value, Some(2.0));
        assert_eq!(result.width, Some(0.0));
    }

    #[test]
    fn time_series_binary_ops_align_on_the_union_of_timestamps() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("price".into(), "A".into()),
            time_input(vec![1.0, 3.0], vec![10.0, 30.0]),
        );
        fetched.insert(
            ("price".into(), "B".into()),
            time_input(vec![2.0, 3.0], vec![20.0, 3.0]),
        );
        let result = replay(
            "show sum(price(\"A\") + price(\"B\"))",
            &fetched,
            &test_params(),
            &TestKernel,
        );
        assert!(result.ok, "{result:?}");
        assert_eq!(result.base_value, Some(33.0));
    }

    #[test]
    fn keyed_table_binary_ops_align_on_the_union_of_row_keys() {
        let mut fetched = BTreeMap::new();
        fetched.insert(
            ("table".into(), "LEFT".into()),
            table_input(&["a", "c"], vec![10.0, 30.0]),
        );
        fetched.insert(
            ("table".into(), "RIGHT".into()),
            table_input(&["b", "c"], vec![20.0, 3.0]),
        );
        let result = replay(
            "show sum(table(\"LEFT\") + table(\"RIGHT\"))",
            &fetched,
            &test_params(),
            &TestKernel,
        );
        assert!(result.ok, "{result:?}");
        assert_eq!(result.base_value, Some(33.0));
    }

    #[test]
    fn ema_matches_pandas_adjust_false_across_missing_rows() {
        let actual = ema(&[f64::NAN, 1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 5.0], 2);
        let expected_bits = [
            0x7ff8_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x4005_b6db_6db6_db6e,
            0x4005_b6db_6db6_db6e,
            0x4005_b6db_6db6_db6e,
            0x4013_84cf_e133_f84c,
        ];
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_bits,
        );
    }
}
