"""Utilities for performing object synchronization."""

import pandas as pd
from unidecode import unidecode
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def string_ngrams(input: str, n: int = 3) -> list[str]:
    """
    Splits a given string into n-character n-grams to use later in cosine similarity.

    Args:
        input (str): any string.
        n (int, optional): the number of characters to use in each n-gram.

    Returns:
        A list of strings of length n.
    """
    if input is None:
        return []

    input = re.sub(r"[,-./]|\s", r"", str(input))
    ngrams = zip(*[input[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


def string_remove_accents(input: str) -> str:
    """
    Uses `unidecode` to convert `input` (a Unicode object/string) into an ASCII-compliant string.

    Please see [`unidecode.unidecode`](https://github.com/takluyver/Unidecode) for more details.

    Args:
        input (str): any Unicode-compliant string.

    Returns:
        A string with only ASCII-compliant characters.
    """
    if input is None:
        return None

    return unidecode(input)


def string_clean_spaces(input: str) -> str:
    """
    Replaces Unicode character U+00A0 (the no-break space) with a "true" space (Unicode character U+0020).

    Args:
        input (str): any string.

    Returns:
        A string with only "true" spaces (U+0020).
    """
    if input is None:
        return None

    return input.replace(" ", " ")


def string_replace_common_womens_suffixes(input: str) -> str:
    """
    Removes common women's club suffixes with empty strings.

    Args:
        input (str): any string.

    Returns:
        A cleaned string without specific text indicating women's teams.
    """
    if input is None:
        return None

    input = input.strip()
    input = re.sub(r",?\s+Women's$", "", input)
    input = re.sub(r",?\s+Women$", "", input)
    input = re.sub(r",?\s+W$", "", input)
    input = re.sub(r"\s+WFC$", "", input)
    input = re.sub(r"\s+LFC$", "", input)
    input = re.sub(r"\s+Ladies$", "", input)
    input = re.sub(r"\s+F$", "", input)
    return (
        input.replace(", Women", "")
        .replace(", Women's", "")
        .replace(" Women's", "")
        .replace(" WFC", "")
        .replace(" Femenino", "")
        .replace(" Femminile", "")
        .replace("Féminas", "")
        .strip()
    )


def string_remove_youth_suffixes(input: str) -> str:
    """
    Removes common youth team suffixes with empty strings.

    Args:
        input (str): any string.

    Returns:
        A cleaned string without specific text indicating youth teams.
    """
    if input is None:
        return None

    input = re.sub(r" Under-?", " U", input)
    input = re.sub(r" Sub-?", " U", input)
    input = re.sub(r" Under ", " U", input)
    input = re.sub(r" U-", " U", input)
    input = re.sub(r" U\s?\d+$", "", input)
    return input.strip()


def series_remove_accents(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Please see [`string_remove_accents`][glass_onion.utils.string_remove_accents] for more details.

    Args:
        input (pd.Series[str]): a pandas.Series with Unicode-compliant strings.

    Returns:
        A pandas.Series with ASCII strings.
    """
    return input.apply(string_remove_accents)


def series_remove_non_word_chars(input: pd.Series) -> "pd.Series[str]":
    """
    Replaces any consecutive punctuation/whitespace/etc. character with one space character.

    Args:
        input (pd.Series[str]): a pandas.Series of strings.

    Returns:
        A pandas.Series of strings.
    """
    return input.str.replace(r"[\W_]+", " ", regex=True)


def series_remove_double_spaces(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Replaces consecutive whitespace characters with just one space character.

    Args:
        input (pd.Series[str]): a pandas.Series of strings.

    Returns:
        A pandas.Series of strings.
    """
    return input.str.replace(r"\s+", " ", regex=True)


def series_clean_spaces(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Please see [`string_clean_spaces`][glass_onion.utils.string_clean_spaces] for more details.

    Args:
        input (pd.Series[str]): a pandas.Series of strings.

    Returns:
        A pandas.Series of strings with only "true" spaces (U+0020).
    """
    return input.apply(string_clean_spaces)


def series_remove_common_suffixes(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Replaces common team suffixes with empty strings.

    Please see [`string_replace_common_womens_suffixes`][glass_onion.utils.string_replace_common_womens_suffixes] and [`string_remove_youth_suffixes`][glass_onion.utils.string_remove_youth_suffixes] for more details.

    Args:
        input (pd.Series[str]): a pandas.Series with club names.

    Returns:
        A pandas.Series with more standardized club names.
    """
    return (
        input.apply(string_replace_common_womens_suffixes)
        .apply(string_remove_youth_suffixes)
        .str.replace(
            r" SC$| Sc$| sc$| FC$| fc$| Fc$| LFC$| CF$| CD$| WFC$| FCW$| HSC$| AC$| AF$| FCO$| Ladies$| Women$| W$|\sW$|, W$| F$| Women\'s$| VF$| FF$| Football$",
            "",
            regex=True,
        )
    )


def series_remove_common_prefixes(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Replaces common team prefixes with empty strings.

    Args:
        input (pd.Series[str]): a pandas.Series with club names.

    Returns:
        A pandas.Series with more standardized club names.
    """
    return input.str.replace(
        r"^SC |^FC |^CF |^CD |^RC |^OL |^Olympique de |^Olympique |^WNT |^SKN |^SK |^1\. ",
        "",
        regex=True,
    )


def series_remove_youth_prefixes(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Replaces common youth team suffixes with empty strings.

    Please see [`string_remove_youth_suffixes`][glass_onion.utils.string_remove_youth_suffixes] for more details.

    Args:
        input (pd.Series[str]): a pandas.Series with club names.

    Returns:
        A pandas.Series with more standardized club names.
    """
    return input.apply(string_remove_youth_suffixes)


def series_normalize(input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Applies a full suite of normalizations to a pandas.Series of strings.

    Please see the following methods for more details:

    * [`series_clean_spaces`][glass_onion.utils.series_clean_spaces]
    * [`series_remove_accents`][glass_onion.utils.series_remove_accents]
    * [`series_remove_non_word_chars`][glass_onion.utils.series_remove_non_word_chars]
    * [`series_remove_double_spaces`][glass_onion.utils.series_remove_double_spaces]

    Args:
        input (pd.Series[str]): a pandas.Series of strings.

    Returns:
        A pandas.Series with normalized strings.
    """
    result = series_clean_spaces(input)
    result = series_remove_accents(result)
    result = series_remove_non_word_chars(result)
    result = series_remove_double_spaces(result)
    result = result.str.lower().str.strip()
    return result


def series_normalize_team_names(self, input: "pd.Series[str]") -> "pd.Series[str]":
    """
    Applies a full suite of normalizations to a pandas.Series of team name strings.

    Please see the following methods for more details:

    * [`series_remove_common_suffixes`][glass_onion.utils.series_remove_common_suffixes]
    * [`series_remove_common_prefixes`][glass_onion.utils.series_remove_common_prefixes]
    * [`series_normalize`][glass_onion.utils.series_normalize]

    Returns:
        A pandas.Series with more standardized club names.
    """
    result = series_remove_common_suffixes(input)
    result = series_remove_common_prefixes(result)
    result = series_normalize(result)
    result = result.str.lower().str.strip()
    return result


def apply_cosine_similarity(
    input1: "pd.Series[str]", input2: "pd.Series[str]"
) -> pd.DataFrame:
    """
    Generates a dataframe of cosine similarity results from two pandas.Series.

    For more technical details on cosine similarity, please see [`sklearn.feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [`sklearn.metrics.pairwise.cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

    The methodology behind this implementation can be found at: https://unravelsports.com/post.html?id=2022-07-11-player-id-matching-system

    Args:
        input1 (pd.Series[str]): a pandas.Series of strings.
        input2 (pd.Series[str]): a pandas.Series of strings.

    Returns:
        pandas.DataFrame: a DataFrame with the schema

            * input1: a string from the `input1` pandas.Series.
            * input1_normalized: the normalized version of the `input1` column.
            * input2: a string from the `input2` pandas.Series.
            * input2_normalized: the normalized version of the `input2` column.
            * similarity (double/float): the cosine similarity score of the normalized strings.
    """
    input1_norm = series_normalize(input1).to_list()
    input2_norm = series_normalize(input2).to_list()

    content = pd.Series(input1_norm + input2_norm).to_list()
    vectorizer = TfidfVectorizer(
        min_df=1, analyzer=string_ngrams, strip_accents="ascii"
    )
    vectorizer.fit(content)  # fit the vectorizer on all teams

    tfidf_i1 = vectorizer.transform(input1_norm)
    tfidf_i2 = vectorizer.transform(input2_norm)

    cosine_sim_matrix = cosine_similarity(tfidf_i2, tfidf_i1)

    row_idx, col_idx = linear_sum_assignment(
        cost_matrix=cosine_sim_matrix, maximize=True
    )

    match_results = []
    for r, c in zip(row_idx, col_idx):
        i1_raw = input1.loc[input1.index[c]]
        i2_raw = input2.loc[input2.index[r]]

        i1_norm = input1_norm[c]
        i2_norm = input2_norm[r]

        similarity = cosine_sim_matrix[r][c]
        match_results.append(
            {
                "input1": i1_raw,
                "input1_normalized": i1_norm,
                "input2": i2_raw,
                "input2_normalized": i2_norm,
                "similarity": similarity,
            }
        )

    return pd.DataFrame(match_results)
