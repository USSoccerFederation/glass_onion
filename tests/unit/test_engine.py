from random import Random
from typing import Tuple
from glass_onion import SyncableContent, SyncEngine
import pytest
import re
import pandas as pd
from glass_onion.utils import series_normalize


def test_init_disjoint_object_types():
    content = [
        SyncableContent(
            object_type=k,
            provider=f"provider_{i}",
            data=pd.DataFrame([{f"provider_{i}_{k}_id": pd.NA}]),
        )
        for i, k in enumerate(["object", "object2", "object3"])
    ]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "One or more `SyncableContent` objects in `content` do not match `SyncEngine.object_type`."
        ),
    ):
        SyncEngine(
            object_type="object",
            content=content,
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


def test_init_content_not_list():
    content = "object"

    with pytest.raises(
        AssertionError,
        match=re.escape("`content` must be a list of SyncableContent objects."),
    ):
        SyncEngine(
            object_type="object",
            content=content,
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


def test_init_content_not_list_of_syncablecontent():
    content = ["object", "object2", "object3"]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "One or more objects in `content` are not `SyncableContent` objects."
        ),
    ):
        SyncEngine(
            object_type="object",
            content=content,
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


def test_init_content_empty():
    with pytest.raises(AssertionError, match=re.escape("`content` can not be empty")):
        SyncEngine(
            object_type="object",
            content=[],
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


def test_init_object_type_null():
    content = [
        SyncableContent(
            object_type="object",
            provider=f"provider_{i}",
            data=pd.DataFrame([{f"provider_{i}_object_id": pd.NA}]),
        )
        for i in range(1, 3)
    ]
    with pytest.raises(
        AssertionError, match=re.escape("`object_type` can not be NULL")
    ):
        SyncEngine(
            object_type=None,
            content=content,
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


def test_init_object_type_empty_whitespace():
    content = [
        SyncableContent(
            object_type="object",
            provider=f"provider_{i}",
            data=pd.DataFrame([{f"provider_{i}_object_id": pd.NA}]),
        )
        for i in range(1, 3)
    ]
    with pytest.raises(
        AssertionError,
        match=re.escape("`object_type` can not be empty or just whitespace"),
    ):
        SyncEngine(
            object_type="     ",
            content=content,
            join_columns=["test_object_id"],
        )

        # should never get here
        assert False


@pytest.mark.parametrize(
    "fields, data, expected_error",
    [
        (
            (),
            None,
            "Must provide two columns (one from `input1` and one from `input2`) as `fields`.",
        ),
        (
            ("object2_name", "object_name"),
            None,
            "First element of `fields` must exist in `input1.data`.",
        ),
        (
            ("object_name", "object2_name"),
            None,
            "Second element of `fields` must exist in `input2.data`.",
        ),
        (
            ("object_name", "object_name"),
            pd.DataFrame(
                [
                    {
                        "provider_a_object_id": 1,
                        "provider_b_object_id": 1,
                        "object_name": "A",
                    }
                ]
            ).head(0),
            "Both SyncableContent objects must be non-empty.",
        ),
        (
            ("object_name", "object_name"),
            pd.DataFrame(
                {
                    "provider_a_object_id": range(1, 3),
                    "provider_b_object_id": range(1, 3),
                    "object_name": [pd.NA] * 2,
                }
            ),
            "Both SyncableContent objects must have > 0 non-null elements in `data`.",
        ),
    ],
)
def test_synchronize_with_error_cases(
    fields: Tuple[str, str], data: pd.DataFrame, expected_error: str
):
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame([{"provider_b_object_id": 1, "object_name": "A"}]),
    )

    if data is not None:
        left.data = data
        right.data = data

    engine = SyncEngine("object", [left, right], ["object_name"])
    methods = [
        "synchronize_with_naive_match",
        "synchronize_with_fuzzy_match",
        "synchronize_with_cosine_similarity",
    ]

    for m in methods:
        with pytest.raises(AssertionError, match=re.escape(expected_error)):
            print(f"Testing failure modes for SyncEngine method: {m}")
            getattr(engine, m)(input1=left, input2=right, fields=fields)


@pytest.mark.parametrize(
    "method",
    [
        "synchronize_with_naive_match",
        "synchronize_with_fuzzy_match",
        "synchronize_with_cosine_similarity",
    ],
)
def test_synchronize_sample_mixed_nulls(method: str):
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            {
                "provider_a_object_id": list(range(1, 4)),
                "object_name": ["Test Team 1", "Test Team 2", "Test Team 3"],
            }
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            {
                "provider_b_object_id": list(range(1, 4)),
                "object_name": ["Test Team 1", pd.NA, "Test Team 3"],
            }
        ),
    )

    engine = SyncEngine("object", [left, right], ["object_name"], verbose=True)

    actual = getattr(engine, method)(left, right, ("object_name", "object_name"))

    assert isinstance(actual, pd.DataFrame)
    assert len(actual) > 0
    assert set(["provider_a_object_id", "provider_b_object_id"]) == set(actual.columns)

    target = actual.loc[actual["provider_a_object_id"] == 1, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_b_object_id"] == 1

    target = actual.loc[actual["provider_a_object_id"] == 3, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_b_object_id"] == 3

    assert len(actual[actual["provider_a_object_id"] == 2]) == 0


@pytest.mark.parametrize(
    "method",
    [
        "synchronize_with_naive_match",
        "synchronize_with_fuzzy_match",
        "synchronize_with_cosine_similarity",
    ],
)
def test_synchronize_population_mixed_nulls(method: str):
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            {
                "provider_a_object_id": list(range(1, 4)),
                "object_name": ["Test Team 1", pd.NA, "Test Team 3"],
            }
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            {
                "provider_b_object_id": list(range(1, 4)),
                "object_name": ["Test Team 1", "Test Team 2", "Test Team 3"],
            }
        ),
    )

    engine = SyncEngine("object", [left, right], ["object_name"], verbose=True)

    actual = getattr(engine, method)(left, right, ("object_name", "object_name"))

    assert isinstance(actual, pd.DataFrame)
    assert len(actual) > 0
    assert set(["provider_a_object_id", "provider_b_object_id"]) == set(actual.columns)

    target = actual.loc[actual["provider_a_object_id"] == 1, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_b_object_id"] == 1

    target = actual.loc[actual["provider_a_object_id"] == 3, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_b_object_id"] == 3

    assert len(actual[actual["provider_a_object_id"] == 2]) == 0


@pytest.mark.parametrize(
    "method",
    [
        "synchronize_with_naive_match",
        "synchronize_with_fuzzy_match",
        "synchronize_with_cosine_similarity",
    ],
)
def test_synchronize_should_match_same_name(method: str):
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            {
                "provider_a_object_id": list(range(1, 4)),
                "object_name": ["Test Team 1", "Test Team 2", "Test Team"],
            }
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            {
                "provider_b_object_id": list(range(1, 4)),
                "object_name": ["Test Team", "Test Team", "Test Team"],
            }
        ),
    )

    engine = SyncEngine("object", [left, right], ["object_name"], verbose=True)

    actual = getattr(engine, method)(left, right, ("object_name", "object_name"))

    assert isinstance(actual, pd.DataFrame)
    assert len(actual) > 0
    assert set(["provider_a_object_id", "provider_b_object_id"]) == set(actual.columns)

    target = actual.loc[actual["provider_a_object_id"] == 3, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_b_object_id"] == 1


def _naive_match_reference(
    left: SyncableContent, right: SyncableContent, fields: Tuple[str, str]
) -> pd.DataFrame:
    # brute-force pairwise implementation of the naive match, used as an oracle
    population = left.data.loc[left.data[fields[0]].notna(), :]
    sample = right.data.loc[right.data[fields[1]].notna(), :]
    population_names = series_normalize(population[fields[0]]).tolist()
    sample_names = series_normalize(sample[fields[1]]).tolist()
    population_ids = population[left.id_field].tolist()
    sample_ids = sample[right.id_field].tolist()

    results = []
    name_map: dict[str, str] = {}
    for i, i1_raw in enumerate(population_names):
        for j, i2_raw in enumerate(sample_names):
            if i1_raw not in name_map and i2_raw not in name_map.values():
                if i1_raw == i2_raw:
                    name_map[i1_raw] = i2_raw
                    results.append((population_ids[i], sample_ids[j]))

    for i, i1_raw in enumerate(population_names):
        if i1_raw in name_map:
            continue
        i1_set = set(re.split(r"\s+", i1_raw))
        for j, i2_raw in enumerate(sample_names):
            if i2_raw in name_map.values():
                continue
            i2_set = set(re.split(r"\s+", i2_raw))
            if i2_set <= i1_set or i1_set <= i2_set:
                name_map[i1_raw] = i2_raw
                results.append((population_ids[i], sample_ids[j]))
                break

    return pd.DataFrame(results, columns=[left.id_field, right.id_field])


@pytest.mark.parametrize("seed", range(500))
def test_synchronize_with_naive_match_matches_reference(seed: int):
    rng = Random(seed)
    # small vocabulary so names overlap often; includes accents, casing, and punctuation
    tokens = ["ana", "bo", "cruz", "de", "silva", "li", "jo", "van", "dijk", "ñu", "Mo-Sa"]

    def random_names() -> list:
        names = [
            None
            if rng.random() < 0.05
            else " ".join(rng.choice(tokens) for _ in range(rng.randint(1, 3)))
            for _ in range(rng.randint(1, 40))
        ]
        # guarantee at least one non-null name so the method's input assertions pass
        if all(n is None for n in names):
            names[0] = rng.choice(tokens)
        return names

    left_names = random_names()
    right_names = random_names()
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            {
                "provider_a_object_id": list(range(len(left_names))),
                "object_name": left_names,
            }
        ),
    )
    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            {
                "provider_b_object_id": list(range(len(right_names))),
                "object_name": right_names,
            }
        ),
    )

    engine = SyncEngine("object", [left, right], ["object_name"])
    fields = ("object_name", "object_name")
    actual = engine.synchronize_with_naive_match(left, right, fields)
    expected = _naive_match_reference(left, right, fields)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
