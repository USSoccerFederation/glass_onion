from pathlib import Path
import pandas as pd
from glass_onion.player import PlayerSyncableContent, PlayerSyncEngine
import pytest

from tests.utils import utils_create_syncables

FIXTURE_DATA_PATH = Path(__file__).resolve().parent / "fixtures" / "player"


@pytest.mark.parametrize(
    "file_path, expected_player_ids",
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
def test_player_synchronize(file_path: str, expected_player_ids: dict[str, str]):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / file_path)

    syncables = utils_create_syncables(dataset, "player")
    engine_test = PlayerSyncEngine(syncables, verbose=False)
    result = engine_test.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_player_ids:
        player_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                player_data = player_data[player_data[f"{provider}_player_id"].isna()]
            else:
                player_data = player_data[
                    player_data[f"{provider}_player_id"] == provider_id
                ]

        assert len(player_data) == 1
