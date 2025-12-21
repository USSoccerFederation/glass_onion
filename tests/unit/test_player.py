from pathlib import Path
from typing import Tuple
import pandas as pd
from glass_onion.engine import SyncableContent
from glass_onion.player import (
    PlayerSyncEngine,
    PlayerSyncLayer,
    PlayerSyncSimilarityMethod,
    PlayerSyncableContent,
)
import pytest
from tests.utils import utils_create_syncables, FIXTURE_DATA_PATH


def test_init_missing_columns():
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame([{"provider_a_player_id": 1, "player_name": "A"}]),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame([{"provider_b_player_id": 1, "player_name": "A"}]),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    assert ["player_name"] == engine.join_columns


def test_init_unreliable_columns():
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [{"provider_a_player_id": 1, "player_name": "A", "team_id": pd.NA}]
        ),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [{"provider_b_player_id": 1, "player_name": "A", "team_id": "A"}]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    assert ["player_name"] == engine.join_columns


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
                    "provider_a_player_id": 1,
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": 1,
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
                    "provider_b_player_id": 1,
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": 0,
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
                        "provider_a_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 1,
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 0,
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
                        "provider_a_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 1,
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 0,
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
                        "provider_a_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 1,
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": 1,
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": 0,
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


@pytest.mark.parametrize(
    "file_path, expected_object_ids",
    [
        # base case
        (
            "2025-07-05-usa-mex.csv",
            [{"provider_a": "332705", "provider_b": "12751", "provider_c": "24629"}],
        ),
        # same name twice but two different players
        (
            "2025-07-15-kor-jpn.csv",
            [
                {"provider_a": "508366", "provider_b": "95406", "provider_c": "155852"},
                {"provider_a": "645847", "provider_b": "96073", "provider_c": "73729"},
            ],
        ),
        # women's game + unused subs + spotty birth dates
        (
            "2024-09-22-chi-sd-w.csv",
            [
                {"provider_a": None, "provider_b": "218500"},  # unused sub, not in A
                {"provider_a": "27889", "provider_b": "33135"},  # naive name matching
                {"provider_a": "13556", "provider_b": "5041"},  # no birth date
            ],
        ),
        # disjoint player sets between data providers
        (
            "2023-09-13-oma-usa.csv",
            [
                {"provider_a": None, "provider_b": "190928"},
                {"provider_a": "429448", "provider_b": None},
            ],
        ),
    ],
)
def test_synchronize_complex_cases(file_path: str, expected_object_ids: dict[str, str]):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / "player" / file_path)

    syncables = utils_create_syncables(dataset, "player")
    engine_test = PlayerSyncEngine(syncables, verbose=False)

    result = engine_test.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_object_ids:
        player_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                player_data = player_data[player_data[f"{provider}_player_id"].isna()]
            else:
                player_data = player_data[
                    player_data[f"{provider}_player_id"] == provider_id
                ]

        assert len(player_data) == 1
