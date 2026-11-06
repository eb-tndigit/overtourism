"""
Safe evaluator for the ``formula:`` expressions of the indicator YAML files.

A formula is a small arithmetic expression over the *names of the phenomena* of an indicator, e.g.::

    beds / population
    sum(presenze_vodafone, presenze_escursionisti_vodafone) / superficie
    coalesce(presenze_vodafone / (presenze_alb + presenze_xalb), presenze_vodafone)

It is parsed once with :mod:`ast` and only a whitelist of syntax is accepted (numbers, phenomenon names,
``+ - * / **``, unary minus, parentheses and the functions in :data:`FUNCTIONS`): nothing else is ever executed, so
a YAML file cannot run arbitrary code. Anything that needs conditions, loops or I/O belongs in a Python combinator
(see ``combinators.py``).

Null semantics (they are the ones the hand-written combinators of the project already had)
-------------------------------------------------------------------------------------------
* ``+``, ``-``, ``*``, ``**``: a null operand gives a null result.
* ``/`` is **zero-safe**: a division by zero is a null (never ``inf``), like a division by a missing value.
* ``sum(a, b, ...)`` adds the values that are present (a missing one counts as 0); it is null only when *all*
  of them are missing.
* ``coalesce(a, b, ...)`` is the first value that is present: use it to give a missing input an explicit value,
  e.g. ``coalesce(ee_services, 0)``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


class FormulaError(ValueError):
    """The formula is not valid (syntax, unsupported construct, unknown function, ...)."""


# ---------------------------------------------------------------------------
# Operators and functions
# ---------------------------------------------------------------------------


def _div(left, right):
    """Zero-safe division: x / 0 is null, not inf."""
    if isinstance(right, pd.Series):
        return left / right.where(right != 0)
    return left / (np.nan if right == 0 else right)


_BINARY_OPERATORS: dict[type, Callable] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _div,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPERATORS: dict[type, Callable] = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}


def _frame(args: Iterable, index: pd.Index) -> pd.DataFrame:
    """The arguments side by side, one column each (scalars are repeated on every row)."""
    columns = {
        i: (
            arg
            if isinstance(arg, pd.Series)
            else pd.Series(arg, index=index, dtype=float)
        )
        for i, arg in enumerate(args)
    }
    return pd.DataFrame(columns, index=index)


def _sum(index, *args):
    return _frame(args, index).sum(axis=1, min_count=1)


def _mean(index, *args):
    return _frame(args, index).mean(axis=1)


def _min(index, *args):
    return _frame(args, index).min(axis=1)


def _max(index, *args):
    return _frame(args, index).max(axis=1)


def _coalesce(index, *args):
    result = args[0]
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=index, dtype=float)
    for fallback in args[1:]:
        result = result.fillna(fallback)
    return result


def _clip(index, value, lower, upper):
    return _frame([value], index)[0].clip(lower=lower, upper=upper)


def _abs(index, value):
    return _frame([value], index)[0].abs()


def _sqrt(index, value):
    series = _frame([value], index)[0]
    return np.sqrt(series.where(series >= 0))  # sqrt of a negative number: null


def _log(index, value):
    series = _frame([value], index)[0]
    return np.log(series.where(series > 0))  # log of zero / a negative number: null


# name -> (function(index, *args), minimum number of arguments, maximum or None, one-line help)
FUNCTIONS: dict[str, tuple[Callable, int, int | None, str]] = {
    "sum": (_sum, 1, None, "row sum of the values that are present (null if none is)"),
    "mean": (_mean, 1, None, "row mean of the values that are present"),
    "min": (_min, 1, None, "row minimum of the values that are present"),
    "max": (_max, 1, None, "row maximum of the values that are present"),
    "coalesce": (_coalesce, 2, None, "first value that is present"),
    "clip": (_clip, 3, 3, "clip(x, lower, upper): limit x to [lower, upper]"),
    "abs": (_abs, 1, 1, "absolute value"),
    "sqrt": (_sqrt, 1, 1, "square root (null for negative values)"),
    "log": (_log, 1, 1, "natural logarithm (null for values <= 0)"),
}


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------


class Formula:
    """
    A parsed, validated formula. ``Formula(text)`` raises :class:`FormulaError` on anything unsupported.

    ``columns`` is the set of names the formula reads; calling the formula on a DataFrame that has those columns
    returns the resulting float Series (same index as the frame).
    """

    def __init__(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise FormulaError("the formula must be a non-empty string")
        self.text = " ".join(text.split())  # single line: used in repr / logs
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"invalid syntax in {self.text!r}: {exc.msg}") from exc
        self._root = tree.body
        columns: set[str] = set()
        self._validate(self._root, columns)
        self.columns: frozenset[str] = frozenset(columns)

    # -- validation ------------------------------------------------------

    def _validate(self, node: ast.AST, columns: set[str]) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise FormulaError(
                    f"only numbers are allowed as constants, got {node.value!r} in {self.text!r}"
                )
        elif isinstance(node, ast.Name):
            columns.add(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY_OPERATORS:
                raise FormulaError(
                    f"unsupported operator {type(node.op).__name__} in {self.text!r} "
                    "(supported: + - * / **)"
                )
            self._validate(node.left, columns)
            self._validate(node.right, columns)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPERATORS:
                raise FormulaError(
                    f"unsupported unary operator {type(node.op).__name__} in {self.text!r}"
                )
            self._validate(node.operand, columns)
        elif isinstance(node, ast.Call):
            self._validate_call(node, columns)
        else:
            raise FormulaError(
                f"unsupported syntax ({type(node).__name__}) in {self.text!r}. A formula can only contain numbers, "
                f"phenomenon names, + - * / **, parentheses and the functions {sorted(FUNCTIONS)}. "
                "Conditions and comparisons need a Python combinator."
            )

    def _validate_call(self, node: ast.Call, columns: set[str]) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            called = ast.unparse(node.func)
            raise FormulaError(
                f"unknown function {called!r} in {self.text!r}. Available: {sorted(FUNCTIONS)}"
            )
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise FormulaError(
                f"{node.func.id}() takes plain positional arguments only, in {self.text!r}"
            )
        _, minimum, maximum, _ = FUNCTIONS[node.func.id]
        count = len(node.args)
        if count < minimum or (maximum is not None and count > maximum):
            expected = (
                f"exactly {minimum}"
                if minimum == maximum
                else (
                    f"at least {minimum}"
                    if maximum is None
                    else f"{minimum} to {maximum}"
                )
            )
            raise FormulaError(
                f"{node.func.id}() takes {expected} argument(s), got {count}, in {self.text!r}"
            )
        for arg in node.args:
            self._validate(arg, columns)

    # -- evaluation ------------------------------------------------------

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        missing = sorted(self.columns - set(df.columns))
        if missing:
            raise KeyError(
                f"formula {self.text!r} needs the column(s) {missing}, "
                f"available: {list(df.columns)}"
            )
        result = self._eval(self._root, df)
        if not isinstance(result, pd.Series):  # a constant formula
            result = pd.Series(result, index=df.index)
        return result.astype(float)

    def _eval(self, node: ast.AST, df: pd.DataFrame) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return df[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY_OPERATORS[type(node.op)](
                self._eval(node.left, df), self._eval(node.right, df)
            )
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPERATORS[type(node.op)](self._eval(node.operand, df))
        # ast.Call (the only other node _validate lets through)
        function = FUNCTIONS[node.func.id][0]
        return function(df.index, *(self._eval(arg, df) for arg in node.args))

    def as_combinator(self) -> Callable[..., pd.Series]:
        """The formula as an Indicator combinator: ``fn(df, **_extra) -> Series`` (the extras are ignored)."""

        def combinator(df: pd.DataFrame, **_extra) -> pd.Series:
            return self(df)

        combinator.__name__ = f"formula({self.text})"
        combinator.formula = self  # type: ignore[attr-defined]
        return combinator

    def __repr__(self) -> str:
        return f"Formula({self.text!r})"
