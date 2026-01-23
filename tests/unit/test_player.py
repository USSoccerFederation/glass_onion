from pathlib import Path
from typing import Tuple
import pandas as pd
import re
from pandera.errors import SchemaError
from glass_onion.engine import SyncableContent
from glass_onion.player import (
    PlayerSyncEngine,
    PlayerSyncLayer,
    PlayerSyncSimilarityMethod,
    PlayerSyncableContent,
)
import pytest


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-21",
        "01-21-2026",
        "21-01-2026",
        "Jan 21, 2026",
        "January 21, 2026",
        "2026-01-26T00:00Z",
        "2026-01-26T00:00:00Z",
        "2026-01-26T00:00:00.000Z",
    ],
)
def test_init_syncable_content_birth_date_is_valid_format(value: str):
    content = PlayerSyncableContent(
        "provider_a",
        pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "test",
                    "birth_date": value,
                    "team_id": "1",
                }
            ]
        ),
    )

    assert content.validate_data_schema()


def test_init_syncable_content_birth_date_is_not_valid_format():
    with pytest.raises(
        SchemaError,
        match=re.escape("Unknown datetime string format, unable to parse: test"),
    ):
        PlayerSyncableContent(
            "provider_a",
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "test",
                        "birth_date": "test",
                        "team_id": "1",
                    }
                ]
            ),
        )


@pytest.mark.parametrize(
    "column",
    [
        "provider_a_player_id",
        "player_name",
        "team_id",
    ],
)
def test_init_syncable_content_prevent_mixed_values(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = pd.NA

        dataset.append(c)

    df = pd.DataFrame(dataset)

    with pytest.raises(
        SchemaError,
        match=re.escape(f"non-nullable series '{column}' contains null values"),
    ):
        PlayerSyncableContent(
            "provider_a",
            df,
        )


@pytest.mark.parametrize(
    "column",
    [
        "player_nickname",
        "birth_date",
        "jersey_number",
    ],
)
def test_init_syncable_content_allow_mixed_values(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = pd.NA

        dataset.append(c)

    df = pd.DataFrame(dataset)

    c = PlayerSyncableContent(
        "provider_a",
        df,
    )
    assert c.validate_data_schema()


def test_init_missing_columns():
    with pytest.raises(
        SchemaError,
        match=re.escape(
            "column 'team_id' not in dataframe. Columns in dataframe: ['provider_a_player_id', 'player_name']"
        ),
    ):
        PlayerSyncableContent(
            "provider_a",
            data=pd.DataFrame([{"provider_a_player_id": "1", "player_name": "A"}]),
        )


def test_init_unreliable_columns():
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "A",
                    "team_id": "A",
                    "jersey_number": pd.NA,
                }
            ]
        ),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "1",
                    "player_name": "A",
                    "team_id": "A",
                    "jersey_number": "1",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    assert set(["player_name", "team_id"]) == set(engine.join_columns)


@pytest.mark.parametrize(
    "layer, method, expected_matches",
    [
        # base case: ensure string similarity hits
        (
            PlayerSyncLayer(
                title="test layer",
                match_methodology=PlayerSyncSimilarityMethod.NAIVE,
                threshold=0,
            ),
            "synchronize_with_naive_match",
            1,
        ),
        # what if someone passes NULL to string similarity?
        (
            PlayerSyncLayer(
                title="test layer",
                match_methodology=None,
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we use different string fields? This won't match in a small sample
        (
            PlayerSyncLayer(
                title="test layer", input_fields=("player_name", "player_nickname")
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we use different string fields AND drop the threshold? this will now match
        (
            PlayerSyncLayer(
                title="test layer",
                input_fields=("player_name", "player_nickname"),
                threshold=0,
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we require certain fields to be equal? This shouldn't match because jersey numbers are different
        (
            PlayerSyncLayer(
                title="test layer", other_equal_fields=["team_id", "jersey_number"]
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we adjust the birth date one day forward? This shouldn't find any matches because the default other_equal_fields includes birth_date
        (
            PlayerSyncLayer(
                title="test layer",
                date_adjustment=pd.Timedelta(days=1),
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we adjust the birth date one day forward AND ignore birth_date? This should find matches
        (
            PlayerSyncLayer(
                title="test layer",
                date_adjustment=pd.Timedelta(days=1),
                other_equal_fields=["team_id"],
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we adjust the birth date one day forward AND swap the month/day? This shouldn't find any matches because the default other_equal_fields includes birth_date
        (
            PlayerSyncLayer(title="test layer", swap_birth_month_day=True),
            "synchronize_with_cosine_similarity",
            0,
        ),
    ],
)
def test_synchronize_using_layer(
    layer: PlayerSyncLayer, method: str, expected_matches: int, mocker
):
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": "1",
                    "birth_date": "1970-01-02",
                }
            ]
        ),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "1",
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": "0",
                    "birth_date": "1970-01-02",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, method)

    result = engine.synchronize_using_layer(left, right, layer)
    if method == "synchronize_with_naive_match":
        spy.assert_called_once_with(left, right, layer.input_fields)
    else:
        spy.assert_called_once_with(
            left, right, layer.input_fields, layer.similarity_threshold
        )
    assert set(["provider_a_player_id", "provider_b_player_id"]) == set(result.columns)

    assert len(result) == expected_matches


@pytest.mark.parametrize(
    "left_df, right_df, remove_columns, expected_layers",
    [
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            [],
            3,
        ),
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            ["jersey_number"],
            1,
        ),
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            ["birth_date"],
            2,
        ),
    ],
)
def test_synchronize_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    remove_columns: list[str],
    expected_layers: int,
    mocker,
):
    left = PlayerSyncableContent(
        "provider_a",
        data=left_df,
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=right_df,
    )

    if len(remove_columns) > 0:
        left.data.drop(remove_columns, axis=1, inplace=True)
        right.data.drop(remove_columns, axis=1, inplace=True)

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, "synchronize_using_layer")

    result = engine.synchronize_pair(left, right)
    assert isinstance(result, PlayerSyncableContent)
    assert len(result.data) == 1
    assert spy.call_count == expected_layers
