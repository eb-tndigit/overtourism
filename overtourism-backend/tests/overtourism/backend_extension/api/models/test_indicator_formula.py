"""Tests of the formula engine used by the `formula:` key of the indicator YAML files."""

import numpy as np
import pandas as pd
import pytest
from overtourism.overtourism.backend_extension.api.models.formula import (
    Formula,
    FormulaError,
)

nan = np.nan


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, nan, 4, 0.0],
            "b": [2, 0, 3, nan, 0.0],
            "c": [nan, nan, 1, 5, 6],
        }
    )


def run(text, df):
    return Formula(text)(df).tolist()


def same(actual, expected):
    np.testing.assert_array_equal(
        np.array(actual, dtype=float), np.array(expected, dtype=float)
    )


def test_division_by_zero_or_null_is_null_never_inf(df):
    same(run("a / b", df), [0.5, nan, nan, nan, nan])
    same(run("a / 0", df), [nan] * 5)


def test_arithmetic_propagates_nulls(df):
    same(run("a + b", df), [3, 2, nan, nan, 0])
    same(run("(a + b) * 2 - 1", df), [5, 3, nan, nan, -1])


def test_sum_ignores_missing_values_but_is_null_when_all_are_missing():
    frame = pd.DataFrame({"a": [1, nan, nan], "b": [2, 3, nan]})
    same(run("sum(a, b)", frame), [3, 3, nan])


def test_coalesce_gives_missing_values_an_explicit_value(df):
    same(run("coalesce(a, 0) / b", df), [0.5, nan, 0, nan, nan])
    same(run("coalesce(a / b, a)", df), [0.5, 2, nan, 4, 0])


def test_other_functions(df):
    same(run("clip(a, 1, 2)", df), [1, 2, nan, 2, 1])
    same(run("max(a, b)", df), [2, 2, 3, 4, 0])
    same(run("min(a, b)", df), [1, 0, 3, 4, 0])
    same(run("mean(a, b)", df), [1.5, 1, 3, 4, 0])
    same(run("sqrt(a)", df)[:2], [1, 2**0.5])
    same(run("log(a)", df), [0, np.log(2), nan, np.log(4), nan])
    same(run("abs(0 - a)", df), [1, 2, nan, 4, 0])


def test_a_constant_formula_is_broadcast_to_every_row(df):
    same(run("5", df), [5] * 5)


def test_columns_used_are_reported():
    assert Formula("sum(a, b) / (c + 1)").columns == {"a", "b", "c"}


def test_missing_column_names_the_available_ones(df):
    with pytest.raises(KeyError, match="zzz"):
        Formula("a / zzz")(df)


@pytest.mark.parametrize(
    "text",
    [
        "a > b",  # comparison
        "a if b else c",  # condition
        "__import__('os')",  # function not in the whitelist
        "a.real",  # attribute
        "a[0]",  # subscript
        "[a]",
        "lambda: 1",
        "'text'",
        "True",
        "foo(a)",
        "sum(a, x=1)",  # keyword argument
        "clip(a, 1)",  # wrong number of arguments
        "a // b",
        "a % b",
        "a +",  # syntax error
        "",
    ],
)
def test_anything_but_plain_arithmetic_is_rejected(text):
    with pytest.raises(FormulaError):
        Formula(text)
