from pathlib import Path
import pandas as pd
from glass_onion.match import MatchSyncEngine
from glass_onion.player import PlayerSyncEngine
import pytest

from glass_onion.team import TeamSyncEngine
from tests.utils import utils_create_syncables

FIXTURE_DATA_PATH = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize(
    "file_path, data_type, expected_object_ids",
    [
        # base case
        (
            "2025-07-05-usa-mex.csv",
            "player",
            [{"provider_a": "332705", "provider_b": "12751", "provider_c": "24629"}],
        ),
        # same name twice but two different players
        (
            "2025-07-15-kor-jpn.csv",
            "player",
            [
                {"provider_a": "508366", "provider_b": "95406", "provider_c": "155852"},
                {"provider_a": "645847", "provider_b": "96073", "provider_c": "73729"},
            ],
        ),
        # women's game + unused subs + spotty birth dates
        (
            "2024-09-22-chi-sd-w.csv",
            "player",
            [
                {"provider_a": None, "provider_b": "218500"},  # unused sub, not in A
                {"provider_a": "27889", "provider_b": "33135"},  # naive name matching
                {"provider_a": "13556", "provider_b": "5041"},  # no birth date
            ],
        ),
        # disjoint player sets between data providers
        (
            "2023-09-13-oma-usa.csv",
            "player",
            [
                {"provider_a": None, "provider_b": "190928"},
                {"provider_a": "429448", "provider_b": None},
            ],
        ),
    ],
)
def test_synchronize(file_path: str, data_type: str, expected_object_ids: dict[str, str]):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / data_type / file_path)

    syncables = utils_create_syncables(dataset, data_type)
    if data_type == "player":
        engine_test = PlayerSyncEngine(syncables, verbose=False)
    elif data_type == "match":
        engine_test = MatchSyncEngine(syncables, verbose=False)
    elif data_type == "team":
        engine_test = TeamSyncEngine(syncables, verbose=False)
    else:
        raise NotImplementedError(f"SyncEngine subclass not implemented for data_type '{data_type}'")

    result = engine_test.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_object_ids:
        player_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                player_data = player_data[player_data[f"{provider}_{data_type}_id"].isna()]
            else:
                player_data = player_data[
                    player_data[f"{provider}_{data_type}_id"] == provider_id
                ]

        assert len(player_data) == 1
