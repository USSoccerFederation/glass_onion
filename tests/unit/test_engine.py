from typing import Tuple
from glass_onion import SyncableContent, SyncEngine
import pytest
import re
import pandas as pd


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


# def test_synchronize_methods_happy_path():
#     # Edge cases
#     #   - same string <-> same string matches properly
#     #   - same string <-> reverse order matches properly
#     #   - Same string <-> random order
#     #   - maiden name <-> married name
#     #   - arabic anglicization
#     #   - cyrillic anglicization
#     #   - ridiculous thresholds
