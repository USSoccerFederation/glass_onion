from pathlib import Path
import pandas as pd
from glass_onion.match import MatchSyncEngine
from glass_onion.player import PlayerSyncEngine
import pytest

from glass_onion.team import TeamSyncEngine
from tests.utils import utils_create_syncables

FIXTURE_DATA_PATH = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.parametrize(
    "file_path, object_type, expected_object_ids",
    [
        ## player
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
        ## match
        # base case + rescheduled
        (
            "2025-mls.csv",
            "match",
            [
                {
                    "provider_a": "3981151",
                    "provider_b": "4513981",
                    "provider_c": "2004931",
                }
            ],
        ),
        # coverage differences
        (
            "2024-25-ucl.csv",
            "match",
            [{"provider_a": "3945546", "provider_b": None}],
        ),
        ## team
        # base case
        (
            "2025-nwsl.csv",
            "team",
            [{"provider_a": "21983", "provider_b": "13449", "provider_c": "3485"}],
        ),
        # coverage differences
        (
            "2024-25-ucl.csv",
            "team",
            [
                {"provider_a": "1028", "provider_b": "197"},
                {"provider_a": "957", "provider_b": None},
            ],
        ),
        # USA games
        (
            "2022-25-usa.csv",
            "match",
            [
                {"provider_a": "3961387", "provider_b": "4447768", "provider_c": "1702326"},
                {"provider_a": "3939974", "provider_b": None, "provider_c":	"1586695"}
            ],
        ),
    ],
)
def test_synchronize(
    file_path: str, object_type: str, expected_object_ids: dict[str, str]
):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / object_type / file_path)

    syncables = utils_create_syncables(dataset, object_type)
    if object_type == "player":
        engine_test = PlayerSyncEngine(syncables, verbose=False)
    elif object_type == "match":
        engine_test = MatchSyncEngine(syncables, verbose=False)
    elif object_type == "team":
        engine_test = TeamSyncEngine(syncables, verbose=False)
    else:
        raise NotImplementedError(
            f"SyncEngine subclass not implemented for object_type '{object_type}'"
        )

    result = engine_test.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_object_ids:
        object_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                object_data = object_data[
                    object_data[f"{provider}_{object_type}_id"].isna()
                ]
            else:
                object_data = object_data[
                    object_data[f"{provider}_{object_type}_id"] == provider_id
                ]

        assert len(object_data) == 1
