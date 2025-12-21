from pathlib import Path
import glass_onion
from glass_onion import string_ngrams
import pytest
import re
import pandas as pd
from pandas.testing import assert_series_equal
from pandas.core.dtypes.common import is_string_dtype, is_float_dtype
from glass_onion.utils import (
    apply_cosine_similarity,
    string_clean_spaces,
    string_remove_accents,
    string_remove_youth_suffixes,
    string_replace_common_womens_suffixes,
)


@pytest.mark.parametrize(
    "input, n, expected",
    [
        (None, 3, []),
        ("test", 3, ["tes", "est"]),
        ("Test;Test", 4, ["Test", "estT", "stTe", "tTes", "Test"]),
        ("Test Test", 4, ["Test", "estT", "stTe", "tTes", "Test"]),
    ],
)
def test_string_ngrams_happy_path(input: str, n: int, expected: list[str]):
    actual = string_ngrams(input, n)
    assert isinstance(actual, list)
    assert actual == expected


def test_string_ngrams_error_n_zero():
    with pytest.raises(
        AssertionError, match=re.escape("Length of n-grams `n` must be greater than 0.")
    ):
        string_ngrams("test", 0)


@pytest.mark.parametrize(
    "method",
    [
        "string_remove_accents",
        "string_clean_spaces",
        "string_replace_common_womens_suffixes",
        "string_remove_youth_suffixes",
    ],
)
def test_string_manipulation_null_returns_null(method: str):
    assert getattr(glass_onion, method)(None) == None, (
        f"Utils method {method} did not return NULL when passed NULL"
    )


@pytest.mark.parametrize(
    "method",
    [
        "string_remove_accents",
        "string_clean_spaces",
        "string_replace_common_womens_suffixes",
        "string_remove_youth_suffixes",
    ],
)
def test_string_manipulation_NA_returns_null(method: str):
    assert getattr(glass_onion, method)(pd.NA) == None, (
        f"Utils method {method} did not return NULL when passed NULL"
    )


@pytest.mark.parametrize(
    "method, expected",
    [
        ("string_remove_accents", "Atlanta Beat  WFC  Under-21"),
        ("string_clean_spaces", "Átlanta Beat  WFC  Under-21"),
        ("string_replace_common_womens_suffixes", "Átlanta Beat   Under-21"),
        ("string_remove_youth_suffixes", "Átlanta Beat  WFC"),
    ],
)
def test_string_manipulation_omnibus(method: str, expected: str):
    actual = getattr(glass_onion, method)("  Átlanta Beat  WFC  Under-21  ")
    assert actual == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (" ", ""),
        ("   ", ""),
        ("  Átlanta Beat  WFC  Under-21  ", "Atlanta Beat  WFC  Under-21"),
        ("Átlanta Beat  WFC", "Atlanta Beat  WFC"),
        ("Atlanta", "Atlanta"),
    ],
)
def test_string_remove_accents(input: str, expected: str):
    actual = string_remove_accents(input)
    assert actual == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (" ", ""),
        ("       ", ""),
        ("Venezuela (Bolivaran Republic)", "Venezuela (Bolivaran Republic)"),
        ("Venezuela   (Bolivaran Republic)", "Venezuela   (Bolivaran Republic)"),
    ],
)
def test_string_clean_spaces(input: str, expected: str):
    actual = string_clean_spaces(input)
    assert actual == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("Atlanta Beat WFC Under-21", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC Under-21  ", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC U 21", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC U-21", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC Sub-21", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC Sub 21", "Atlanta Beat WFC"),
        ("Atlanta Beat WFC", "Atlanta Beat WFC"),
        ("Atlanta Beat Sub-21 WFC", "Atlanta Beat U21 WFC"),
    ],
)
def test_string_remove_youth_suffixes(input: str, expected: str):
    actual = string_remove_youth_suffixes(input)
    assert actual == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("Atlanta Beat WFC", "Atlanta Beat"),
        ("Atlanta Beat.             WFC", "Atlanta Beat."),
        ("Atlanta Beat Women''s", "Atlanta Beat"),
        ("Atlanta Beat Women's", "Atlanta Beat"),
        ("Atlanta Beat WFC    ", "Atlanta Beat"),
        ("Atlanta Beat Féminas WFC", "Atlanta Beat"),
        ("LFC Atlanta Beat", "LFC Atlanta Beat"),
    ],
)
def test_string_replace_common_womens_suffixes(input: str, expected: str):
    actual = string_replace_common_womens_suffixes(input)
    assert actual == expected


@pytest.mark.parametrize(
    "method",
    [
        "series_remove_accents",
        "series_remove_non_word_chars",
        "series_remove_double_spaces",
        "series_clean_spaces",
        "series_remove_common_suffixes",
        "series_remove_common_prefixes",
        "series_remove_youth_prefixes",
        "series_normalize",
        "series_normalize_team_names",
    ],
)
def test_series_manipulation_empty_series_returns_empty_series(method: str):
    input = pd.Series()
    expected = pd.Series()
    actual = getattr(glass_onion, method)(input)
    assert assert_series_equal(actual, expected) == None


@pytest.mark.parametrize(
    "method",
    [
        "series_remove_accents",
        "series_remove_non_word_chars",
        "series_remove_double_spaces",
        "series_clean_spaces",
        "series_remove_common_suffixes",
        "series_remove_common_prefixes",
        "series_remove_youth_prefixes",
        "series_normalize",
        "series_normalize_team_names",
    ],
)
def test_series_manipulation_null_returns_null(method: str):
    assert getattr(glass_onion, method)(None) == None


@pytest.mark.parametrize(
    "method",
    [
        "series_remove_accents",
        "series_remove_non_word_chars",
        "series_remove_double_spaces",
        "series_clean_spaces",
        "series_remove_common_suffixes",
        "series_remove_common_prefixes",
        "series_remove_youth_prefixes",
        "series_normalize",
        "series_normalize_team_names",
    ],
)
def test_series_manipulation_series_all_nulls_returns_series_all_nulls(method: str):
    input = pd.Series([None] * 10)
    expected = pd.Series([None] * 10)
    actual = getattr(glass_onion, method)(input)
    assert assert_series_equal(actual, expected) == None


@pytest.mark.parametrize(
    "method",
    [
        "series_remove_accents",
        "series_remove_non_word_chars",
        "series_remove_double_spaces",
        "series_clean_spaces",
        "series_remove_common_suffixes",
        "series_remove_common_prefixes",
        "series_remove_youth_prefixes",
        "series_normalize",
        "series_normalize_team_names",
    ],
)
def test_series_manipulation_mixed_nulls_returns_mixed_nulls(method: str):
    input = pd.Series(([None] * 10) + (["test"] * 10))
    expected = pd.Series(([None] * 10) + (["test"] * 10))
    actual = getattr(glass_onion, method)(input)
    assert assert_series_equal(actual, expected) == None


def test_apply_cosine_similarity_happy_path():
    input1 = pd.Series(["Test Team 1", "Test Team 2", "Test Team 3"])
    input2 = pd.Series(["Test Team 1", "Test Team 2", "Test Team 3"])

    actual = apply_cosine_similarity(input1, input2)
    assert isinstance(actual, pd.DataFrame)
    assert len(actual) == 3

    columns = [
        "input1",
        "input1_normalized",
        "input2",
        "input2_normalized",
        "similarity",
    ]
    assert all([(k in actual.columns) for k in columns])
    assert all([is_string_dtype(actual[k]) for k in columns if k != "similarity"])
    assert is_float_dtype(actual["similarity"])

    target = actual.loc[actual["input1"] == "Test Team 1", :]
    assert len(target) == 1
    assert target.loc[target.index[0], "input1_normalized"] == "test team 1"
    assert target.loc[target.index[0], "input2"] == "Test Team 1"
    assert target.loc[target.index[0], "input2_normalized"] == "test team 1"
    assert int(target.loc[target.index[0], "similarity"]) == 1.0


def test_apply_cosine_similarity_mixed_nulls():
    input1 = pd.Series(["Test Team 1", "Test Team 2", "Test Team 3"])
    input2 = pd.Series(["Test Team 1", pd.NA, "Test Team 3"])

    actual = apply_cosine_similarity(input1, input2)
    assert isinstance(actual, pd.DataFrame)
    assert len(actual) == 2

    columns = [
        "input1",
        "input1_normalized",
        "input2",
        "input2_normalized",
        "similarity",
    ]
    assert all([(k in actual.columns) for k in columns])
    assert all([is_string_dtype(actual[k]) for k in columns if k != "similarity"])
    assert is_float_dtype(actual["similarity"])

    target = actual.loc[actual["input1"] == "Test Team 1", :]
    assert len(target) == 1
    assert target.loc[target.index[0], "input1_normalized"] == "test team 1"
    assert target.loc[target.index[0], "input2"] == "Test Team 1"
    assert target.loc[target.index[0], "input2_normalized"] == "test team 1"
    assert int(target.loc[target.index[0], "similarity"]) == 1

    should_be_missing = actual.loc[actual["input1"] == "Test Team 2", :]
    assert len(should_be_missing) == 0


def test_apply_cosine_similarity_error_series_all_nulls():
    input1 = pd.Series([pd.NA] * 10)
    input2 = pd.Series(["Test Team 1", pd.NA, "Test Team 3"])

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Both `input1` and `input2` must include > 0 non-null/NA elements."
        ),
    ):
        apply_cosine_similarity(input1, input2)
