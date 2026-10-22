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
        None,
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
    ["01-21-2026", "21-01-2026", "Jan 21, 2026", "January 21, 2026", "test"],
)
def test_init_syncable_content_birth_date_is_not_valid_format(value: str):
    with pytest.raises(
        SchemaError,
        match=re.escape(
            "Column 'birth_date' failed series or dataframe validator 0: <Check is_valid_iso8601_timestamp>"
        ),
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
        ).validate_data_schema()


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
        ).validate_data_schema()


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
        ).validate_data_schema()


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
        ).validate_data_schema()


def test_init_engine_validates_content():
    # schema validation is deferred from SyncableContent creation to SyncEngine creation
    content = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame([{"provider_a_player_id": "1", "player_name": "A"}]),
    )

    with pytest.raises(
        SchemaError,
        match=re.escape(
            "column 'team_id' not in dataframe. Columns in dataframe: ['provider_a_player_id', 'player_name']"
        ),
    ):
        PlayerSyncEngine([content])


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
    layer: PlayerSyncLayer,
    method: str,
    expected_call_count: int,
    expected_matches: int,
    mocker,
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
        # a 0-day adjustment is only applied to input1, since adjusting input2 by 0 days compares the same dates
        if not (adjusted_input == 2 and days == 0)
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


# ---------------------------------------------------------------------------------------------------------------------
# Known gaps
#
# Each test is a strict xfail until its gap is fixed; once fixed, remove the marker.
# ---------------------------------------------------------------------------------------------------------------------


def gap_content(provider: str, rows: list[tuple], **columns) -> PlayerSyncableContent:
    # rows: (player ID, player_name, team_id); player_nickname defaults to player_name. columns: extra columns, one value
    # per row
    data = pd.DataFrame(
        rows, columns=[f"{provider}_player_id", "player_name", "team_id"]
    )
    data["player_nickname"] = data["player_name"]
    for k, v in columns.items():
        data[k] = v
    return PlayerSyncableContent(provider, data)


def linked(result: pd.DataFrame) -> set[frozenset]:
    # the groups of two or more provider IDs that the engine linked
    id_columns = [c for c in result.columns if c.endswith("_player_id")]
    groups = {
        frozenset(r[c] for c in id_columns if pd.notna(r[c]))
        for r in result.to_dict("records")
    }
    return {g for g in groups if len(g) > 1}


def all_ids(result: pd.DataFrame) -> set:
    id_columns = [c for c in result.columns if c.endswith("_player_id")]
    return {v for c in id_columns for v in result[c] if pd.notna(v)}


def test_gap_matches_within_team_when_similar_names_are_on_both_teams():
    a = gap_content(
        "provider_a", [("a1", "Sam Kerr", "T1"), ("a2", "Samantha Kerr", "T2")]
    )
    b = gap_content(
        "provider_b", [("b1", "Samantha Kerr", "T1"), ("b2", "Sam Kerr", "T2")]
    )
    result = PlayerSyncEngine([a, b]).synchronize().data
    assert linked(result) == {frozenset({"a1", "b1"}), frozenset({"a2", "b2"})}


def test_gap_naive_match_ignores_candidates_on_other_teams():
    a = gap_content("provider_a", [("a1", "Ana Silva", "T1")])
    b = gap_content(
        "provider_b", [("b1", "Silva", "T2"), ("b2", "Ana Maria Silva", "T1")]
    )
    engine = PlayerSyncEngine([a, b])
    layer = PlayerSyncLayer(
        title="naive x team",
        match_methodology=PlayerSyncSimilarityMethod.NAIVE,
        date_adjustment=None,
        other_equal_fields=["team_id"],
    )
    result = engine.synchronize_using_layer(a, b, layer)
    assert linked(result) == {frozenset({"a1", "b2"})}


def test_gap_layer5_joins_on_jersey_and_team():
    a = gap_content(
        "provider_a",
        [("a1", "Xyz Qwerty", "T1"), ("a2", "Abc Def", "T1")],
        jersey_number=["5", "7"],
    )
    b = gap_content(
        "provider_b",
        [("b1", "Abc Dex", "T1"), ("b2", "Zzz Mno", "T1")],
        jersey_number=["5", "7"],
    )
    result = PlayerSyncEngine([a, b]).synchronize().data
    assert linked(result) == {frozenset({"a1", "b1"}), frozenset({"a2", "b2"})}


@pytest.mark.xfail(
    strict=True,
    reason="when two players have the same name, the engine links to one of them by row order instead of treating the "
    "match as ambiguous",
)
def test_gap_does_not_pick_between_players_with_the_same_name():
    # one team holds every player (IE: gender as team_id), team-only matching off
    a = gap_content("provider_a", [("a1", "Sarah Smith", "female")])
    b = gap_content(
        "provider_b", [("b1", "Sarah Smith", "female"), ("b2", "Sarah Smith", "female")]
    )
    result = PlayerSyncEngine([a, b], allow_team_only_layer=False).synchronize().data
    assert not any("a1" in g for g in linked(result)), (
        f"a1 linked by row order: {linked(result)}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="SyncEngine.synchronize groups the result by join_columns (team_id, player_name) and keeps the first ID per "
    "provider, so a second player with the same name loses their ID",
)
def test_gap_keeps_every_id_when_unmatched_players_share_a_name():
    a = gap_content(
        "provider_a", [("a1", "Sarah Smith", "female"), ("a2", "Sarah Smith", "female")]
    )
    b = gap_content("provider_b", [("b1", "Bob Jones", "female")])
    result = PlayerSyncEngine([a, b], allow_team_only_layer=False).synchronize().data
    assert all_ids(result) == {"a1", "a2", "b1"}


def test_gap_result_does_not_depend_on_provider_order():
    # the three pair results contradict each other: a1-b1, b1-c1 and a1-c2
    def content():
        return [
            gap_content("provider_a", [("a1", "Chris Lee", "T1")]),
            gap_content("provider_b", [("b1", "Christopher Lee", "T1")]),
            gap_content(
                "provider_c",
                [("c1", "Christopher Lee", "T1"), ("c2", "Chris Lee", "T1")],
            ),
        ]

    forward_engine = PlayerSyncEngine(content())
    forward = forward_engine.synchronize().data
    backward = PlayerSyncEngine(content()[::-1]).synchronize().data
    assert linked(forward) == linked(backward)
    # the two exact-name Layer 1 links beat the team-only Layer 5 link, which is reported as a conflict
    assert linked(forward) == {frozenset({"a1", "c2"}), frozenset({"b1", "c1"})}
    assert forward_engine.conflicts == [
        (("provider_a_player_id", "a1"), ("provider_b_player_id", "b1"))
    ]


def test_conflicts_survive_the_remainder_pass():
    # the same contradicting links as above, plus a player each in providers A and B on teams the others don't have:
    # they stay unsynced after the first pass, so synchronize() runs a second pass on the remainders
    a = gap_content(
        "provider_a", [("a1", "Chris Lee", "T1"), ("a2", "Lone Striker", "T2")]
    )
    b = gap_content(
        "provider_b",
        [("b1", "Christopher Lee", "T1"), ("b2", "Lone Keeper", "T3")],
    )
    c = gap_content(
        "provider_c", [("c1", "Christopher Lee", "T1"), ("c2", "Chris Lee", "T1")]
    )
    engine = PlayerSyncEngine([a, b, c])
    result = engine.synchronize().data

    assert {"a2", "b2"} <= all_ids(result)
    assert engine.conflicts == [
        (("provider_a_player_id", "a1"), ("provider_b_player_id", "b1"))
    ]


@pytest.mark.parametrize(
    "other_jersey",
    [
        "9",
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="one missing jersey number anywhere marks jersey_number unreliable for every pair, so with team-only "
                "matching off, Layer 5 is skipped even for players whose jersey numbers are all known",
            ),
        ),
    ],
)
def test_gap_one_missing_jersey_does_not_turn_off_jersey_matching(other_jersey):
    # no name test passes for this pair, so only Layer 5 (jersey number x team) can link them
    a = gap_content(
        "provider_a",
        [("a1", "Kateryna Zhytomyrska", "T1"), ("a2", "Lone Substitute", "T2")],
        jersey_number=["1", other_jersey],
    )
    b = gap_content(
        "provider_b", [("b1", "Ekaterina Jitomirskaya", "T1")], jersey_number=["1"]
    )
    result = PlayerSyncEngine([a, b], allow_team_only_layer=False).synchronize().data
    assert frozenset({"a1", "b1"}) in linked(result)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="player_nickname is optional in PlayerDataSchema, but Layers 3 and 4 require it",
)
def test_gap_runs_without_player_nickname():
    a = PlayerSyncableContent(
        "provider_a",
        pd.DataFrame(
            {
                "provider_a_player_id": ["a1"],
                "player_name": ["Jane Doe"],
                "team_id": ["T1"],
            }
        ),
    )
    b = PlayerSyncableContent(
        "provider_b",
        pd.DataFrame(
            {
                "provider_b_player_id": ["b1"],
                "player_name": ["Janet Doe"],
                "team_id": ["T1"],
            }
        ),
    )
    PlayerSyncEngine([a, b]).synchronize()


def test_gap_runs_when_only_one_provider_has_jersey_numbers():
    a = gap_content("provider_a", [("a1", "Jane Doe", "T1")], jersey_number=["4"])
    b = gap_content("provider_b", [("b1", "Jane Doe", "T1")])
    result = PlayerSyncEngine([a, b]).synchronize().data
    assert linked(result) == {frozenset({"a1", "b1"})}


@pytest.mark.parametrize(
    "adjusted_input",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                strict=True,
                reason="only the adjusted input's birth_date is reformatted to YYYY-MM-DD; the other is compared as its raw "
                "string",
            ),
        ),
    ],
)
def test_gap_birth_date_layer_compares_dates_not_strings(adjusted_input):
    a = gap_content(
        "provider_a", [("a1", "Jane Doe", "T1")], birth_date=["1995-03-04 00:00:00"]
    )
    b = gap_content("provider_b", [("b1", "Jane Doe", "T1")], birth_date=["1995-03-04"])
    layer = PlayerSyncLayer(
        title="birth date x team",
        date_adjustment=pd.Timedelta(0),
        date_adjustment_input=adjusted_input,
    )
    result = PlayerSyncEngine([a, b]).synchronize_using_layer(a, b, layer)
    assert linked(result) == {frozenset({"a1", "b1"})}
