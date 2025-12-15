import glass_onion
from glass_onion import string_ngrams
import pytest
import re
import pandas as pd
from pandas.testing import assert_series_equal


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
        "series_remove_accents",
        "series_remove_non_word_chars",
        "series_remove_double_spaces",
        "series_clean_spaces",
        "series_remove_common_suffixes",
        "series_remove_common_prefixes",
        "series_remove_youth_prefixes",
        "series_normalize",
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
    ],
)
def test_series_manipulation_series_all_nulls_returns_series_all_nulls(method: str):
    input = pd.Series([None] * 10)
    expected = pd.Series([None] * 10)
    actual = getattr(glass_onion, method)(input)
    assert assert_series_equal(actual, expected) == None


# @pytest.mark.parametrize(
#     "method",
#     [
#         "series_remove_accents",
#         "series_remove_non_word_chars",
#         "series_remove_double_spaces",
#         "series_clean_spaces",
#         "series_remove_common_suffixes",
#         "series_remove_common_prefixes",
#         "series_remove_youth_prefixes",
#         "series_normalize",
#     ],
# )
# def test_series_manipulation_series_all_nulls_returns_series_all_nulls(method: str):
#     input = pd.Series([None] * 10)
#     expected = pd.Series([None] * 10)
#     actual = getattr(glass_onion, method)(input)
#     assert assert_series_equal(actual, expected) == None
