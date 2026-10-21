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


@pytest.mark.parametrize(
    "value",
    [
        "01-21-2026",
        "21-01-2026",
        "Jan 21, 2026",
        "January 21, 2026",
        "test"
    ],
)
def test_init_syncable_content_birth_date_is_not_valid_format(value: str):
    with pytest.raises(
        SchemaError,
        match=re.escape("Column 'birth_date' failed series or dataframe validator 0: <Check is_valid_iso8601_timestamp>"),
    ):
        PlayerSyncableContent(
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


@pytest.mark.parametrize(
    "column",
    [
        "provider_a_player_id",
        "team_id",
        "jersey_number",
    ],
)
def test_init_syncable_content_allow_mixed_types(column: str):
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
            c[column] = i

        dataset.append(c)

    df = pd.DataFrame(dataset)

    c = PlayerSyncableContent(
        "provider_a",
        df,
    )
    assert c.validate_data_schema()


@pytest.mark.parametrize(
    "column",
    ["player_nickname", "player_name", "birth_date"],
)
def test_init_syncable_content_prevent_mixed_types(column: str):
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
            c[column] = i

        dataset.append(c)

    df = pd.DataFrame(dataset)

    with pytest.raises(
        SchemaError,
        match=re.escape(f"expected series '{column}' to have type str"),
    ):
        PlayerSyncableContent(
            "provider_a",
            df,
        )


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
    "layer, method, expected_call_count, expected_matches",
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
            1,
        ),
        # what if we use different string fields? This won't match in a small sample
        (
            PlayerSyncLayer(
                title="test layer", input_fields=("player_name", "player_nickname")
            ),
            "synchronize_with_cosine_similarity",
            1,
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
            1,
        ),
        # what if we require certain fields to be equal? This shouldn't match because jersey numbers are different
        (
            PlayerSyncLayer(
                title="test layer", other_equal_fields=["team_id", "jersey_number"]
            ),
            "synchronize_with_cosine_similarity",
            0,
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
            1,
        ),
        # what if we adjust the birth date one day forward AND swap the month/day? This shouldn't find any matches because the default other_equal_fields includes birth_date
        (
            PlayerSyncLayer(title="test layer", swap_birth_month_day=True),
            "synchronize_with_cosine_similarity",
            0,
            0,
        ),
    ],
)
def test_synchronize_using_layer(
    layer: PlayerSyncLayer, method: str, expected_call_count: int, expected_matches: int, mocker
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
                    "birth_date": "1971-01-02",
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
                    "birth_date": "1971-01-02",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, method)

    result = engine.synchronize_using_layer(left, right, layer)
    assert spy.call_count == expected_call_count
    if expected_call_count > 0:
        assert spy.call_args_list[0][1]["fields"] == layer.input_fields

        if method == "synchronize_with_cosine_similarity":
            assert spy.call_args_list[0][1]["threshold"] == layer.similarity_threshold

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


def test_synchronize_pair_birth_date_layers_cover_both_inputs(mocker):
    left_player = pd.DataFrame(
        [
            {
                "provider_a_player_id": "1",
                "player_name": "Abby Wambach",
                "player_nickname": "A. Wambach",
                "team_id": "A",
                "jersey_number": "1",
                "birth_date": "1971-01-02",
            }
        ]
    )
    left = PlayerSyncableContent(
        "provider_a",
        data=left_player,
    )

    # different teams, so no layer matches and every strategy is tried
    right_player = left_player.copy(deep=True)
    right_player["team_id"] = "B"
    right_player.rename(
        {"provider_a_player_id": "provider_b_player_id"}, axis=1, inplace=True
    )
    right = PlayerSyncableContent(
        "provider_b",
        data=right_player,
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, "synchronize_using_layer")
    engine.synchronize_pair(left, right)

    layers = [
        c.args[2] for c in spy.call_args_list if c.args[2].title.startswith("Layer 2")
    ]
    expected = {
        (adjusted_input, swap, days, fields)
        for adjusted_input in [1, 2]
        for swap in [False, True]
        for days in [-1, 0, 1]
        for fields in [
            ("player_name", "player_name"),
            ("player_name", "player_nickname"),
            ("player_nickname", "player_name"),
            ("player_nickname", "player_nickname"),
        ]
    }
    actual = [
        (
            l.date_adjustment_input,
            l.swap_birth_month_day,
            l.date_adjustment.days,
            l.input_fields,
        )
        for l in layers
    ]
    assert len(actual) == len(expected)
    assert set(actual) == expected
    # input1's birth dates are adjusted before input2's
    assert [a[0] for a in actual] == sorted(a[0] for a in actual)


@pytest.mark.parametrize(
    "date_adjustment_input, expected_matches",
    [
        # 2001-05-11 + 1 day, with month/day swapped, is 2001-12-05
        (2, 1),
        # no adjustment of 2001-12-05 by +1 day (with or without a swap) gives 2001-05-11
        (1, 0),
    ],
)
def test_synchronize_using_layer_adjusts_chosen_input(
    date_adjustment_input: int, expected_matches: int
):
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "Mia Hamm",
                    "player_nickname": "M. Hamm",
                    "team_id": "A",
                    "jersey_number": "1",
                    "birth_date": "2001-12-05",
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
                    "player_name": "Mia Hamm",
                    "player_nickname": "M. Hamm",
                    "team_id": "A",
                    "jersey_number": "0",
                    "birth_date": "2001-05-11",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    layer = PlayerSyncLayer(
        title="test layer",
        date_adjustment=pd.Timedelta(days=1),
        swap_birth_month_day=True,
        date_adjustment_input=date_adjustment_input,
    )
    result = engine.synchronize_using_layer(left, right, layer)
    assert len(result) == expected_matches


def test_synchronize_pair_matches_on_input2_birth_date_adjustment(mocker):
    # only adjusting input2 (2001-05-11 + 1 day, month/day swapped) gives input1's birth date
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "Carli Lloyd",
                    "player_nickname": "C. Lloyd",
                    "team_id": "A",
                    "jersey_number": "1",
                    "birth_date": "2001-12-05",
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
                    "player_name": "Carli Lloyd",
                    "player_nickname": "C. Lloyd",
                    "team_id": "A",
                    "jersey_number": "0",
                    "birth_date": "2001-05-11",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, "synchronize_using_layer")
    result = engine.synchronize_pair(left, right)

    assert len(result.data) == 1
    matched = [
        c.args[2] for c, r in zip(spy.call_args_list, spy.spy_return_list) if len(r) > 0
    ]
    assert len(matched) == 1
    assert matched[0].title.startswith("Layer 2")
    assert matched[0].date_adjustment_input == 2
    assert matched[0].swap_birth_month_day
    assert matched[0].date_adjustment == pd.Timedelta(days=1)


@pytest.mark.parametrize(
    "allow_team_only_layer, expected_matches",
    [
        (True, 1),
        (False, 0),
    ],
)
def test_synchronize_pair_team_only_layer(
    allow_team_only_layer: bool, expected_matches: int, mocker
):
    # no jersey numbers and names with nothing in common: only a team-only Layer 5 can match these
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "Landon Donovan",
                    "player_nickname": "Landon Donovan",
                    "team_id": "A",
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
                    "player_name": "Clint Dempsey",
                    "player_nickname": "Clint Dempsey",
                    "team_id": "A",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine(
        [left, right], verbose=True, allow_team_only_layer=allow_team_only_layer
    )
    spy = mocker.spy(engine, "synchronize_using_layer")
    result = engine.synchronize_pair(left, right)

    assert len(result.data) == expected_matches
    titles = [c.args[2].title for c in spy.call_args_list]
    assert ("Layer 5: team" in titles) == allow_team_only_layer
