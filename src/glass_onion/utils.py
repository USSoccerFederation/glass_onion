import pandas as pd
from unidecode import unidecode
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def string_ngrams(string, n=3):
    """
    Chop names into 3 letter 'words' to try and make better fits
    """
    string = re.sub(r"[,-./]|\s", r"", str(string))
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


def string_remove_accents(input_str: str):
    return unidecode(input_str)


def series_remove_accents(input: pd.Series):
    return input.apply(string_remove_accents)


def series_remove_dashes(input: pd.Series):
    return input.str.replace(r"[\W_]+", " ", regex=True)


def series_remove_double_spaces(input: pd.Series):
    return input.str.replace(r"\s+", " ", regex=True)


def series_remove_common_suffixes(input: pd.Series):
    return input.str.replace(
        r" SC$| Sc$| sc$| FC$| fc$| Fc$| LFC$| CF$| CD$| WFC$| FCW$| HSC$| AC$| AF$| FCO$| Ladies$| Women$| W$|\sW$|, W$| F$| Women\'s$| VF$| FF$| Football$",
        "",
        regex=True,
    )


def series_remove_common_prefixes(input: pd.Series):
    return input.str.replace(
        r"^SC |^FC |^CF |^CD |^RC |^OL |^Olympique de |^Olympique |^WNT |^SKN |^SK ",
        "",
        regex=True,
    )


def series_remove_youth_prefixes(input: pd.Series):
    return input.str.replace(r" Under-?", " U", regex=True).str.replace(
        r" Under ", " U", regex=True
    )


def series_normalize(input: pd.Series) -> pd.Series:
    result = series_remove_accents(input)
    result = series_remove_dashes(result)
    result = series_remove_double_spaces(result)
    result = result.str.lower().str.strip()
    return result


def apply_cosine_similarity(input1: pd.Series, input2: pd.Series):
    input1_norm = series_normalize(input1).tolist()
    input2_norm = series_normalize(input2).tolist()

    content = pd.Series(input1_norm + input2_norm).tolist()
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
        i1_raw = input1[input1.index[c]]
        i2_raw = input2[input2.index[r]]

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
