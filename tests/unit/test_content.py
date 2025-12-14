from glass_onion import SyncableContent, SyncEngine
import pandas as pd
import pytest
import re


def test_init_dataframe_none():
    with pytest.raises(AssertionError, match=re.escape("Field `data` can not be null")):
        left = SyncableContent("object", "provider_a", data=None)

        # should never get here
        assert False


def test_init_dataframe_missing_id_field():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Field `provider_a_object_id` must be available as a column in `data`"
        ),
    ):
        SyncableContent("object", "provider_a", data=pd.DataFrame())

        # should never get here
        assert False


def test_merge_with_none():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    assert left.merge(None) == left


def test_merge_disjoint_object_types():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    right = SyncableContent(
        "object2",
        "provider_b",
        data=pd.DataFrame([{"provider_b_object2_id": 2, "object_name": "A"}]),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Left `object_type` (object) does not match Right `object_type` (object2)."
        ),
    ):
        left.merge(right)


def test_merge_missing_right_id_column_in_left():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_object_id": 1,
                    "object_name": "A",
                    "provider_c_object_id": 1,
                }
            ]
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame([{"provider_b_object_id": 2, "object_name": "A"}]),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Right `id_field` (provider_b_object_id) not in Left `data` columns."
        ),
    ):
        left.merge(right)


def test_merge_happy_path():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_object_id": 1,
                    "object_name": "A",
                    "provider_b_object_id": 2,
                }
            ]
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_object_id": 2,
                    "object_name": "A",
                    "provider_c_object_id": 3,
                }
            ]
        ),
    )

    merged = left.merge(right)

    # new object
    assert merged != left and merged != right
    # confirm left merge
    assert len(left.data) == len(merged.data)
    # confirm right ID columns added
    assert "provider_c_object_id" in merged.data.columns
    # confirm data merge
    target = merged.data.loc[merged.data["provider_b_object_id"] == 2, :]
    assert len(target) == 1
    assert target.loc[target.index[0], "provider_a_object_id"] == 1
    assert target.loc[target.index[0], "provider_c_object_id"] == 3


def test_append_none():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    assert left.append(None) == left


def test_append_dataframe_empty():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    assert left.append(pd.DataFrame()) == left


def test_append_syncablecontent_none():
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
    right.data = None

    assert left.append(right) == left


def test_append_syncablecontent_disjoint_object_types():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame([{"provider_a_object_id": 1, "object_name": "A"}]),
    )

    right = SyncableContent(
        "object2",
        "provider_b",
        data=pd.DataFrame([{"provider_b_object2_id": 2, "object_name": "A"}]),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Left `object_type` (object) does not match Right `object_type` (object2)."
        ),
    ):
        left.merge(right)


def test_append_happy_path():
    left = SyncableContent(
        "object",
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_object_id": 1,
                    "object_name": "A",
                    "provider_b_object_id": 2,
                }
            ]
        ),
    )

    right = SyncableContent(
        "object",
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_a_object_id": 1,
                    "provider_b_object_id": 2,
                    "object_name": "A",
                    "provider_c_object_id": 3,
                }
            ]
        ),
    )

    appended = left.append(right)

    # confirm in place change
    assert appended == left
    assert len(left.data) == len(appended.data)

    # confirm right ID columns added
    assert "provider_c_object_id" in appended.data.columns

    # confirm data appended
    target = appended.data.loc[appended.data["provider_b_object_id"] == 2, :]
    assert len(target) == 2
