"""
This file is virtually the same as `tests/functional/test_synchronize.py`, but reformatted to allow for use in isolation without `pytest` and any package-specific logic (test utilities, etc).
"""

from pathlib import Path
import pandas as pd
from glass_onion.engine import SyncableContent
from glass_onion.match import MatchSyncEngine
from glass_onion.player import PlayerSyncEngine

from glass_onion.team import TeamSyncEngine

FIXTURE_DATA_PATH = Path(__file__).resolve().parent.parent / "fixtures"


def utils_transform_provider_data(
    dataset: pd.DataFrame, provider: str, object_type: str
) -> pd.DataFrame:
    generic_id = f"provider_{object_type}_id"
    specific_id = f"{provider}_{object_type}_id"

    dataset.rename({generic_id: specific_id}, axis=1, inplace=True)
    dataset[specific_id] = dataset[specific_id].round().astype("Int64").astype(str)
    dataset.drop(["data_provider"], axis=1, inplace=True)
    return dataset


def utils_create_syncables(
    dataset: pd.DataFrame, object_type: str
) -> list[SyncableContent]:
    grouped = dataset.groupby("data_provider")
    syncables = [
        SyncableContent(
            provider=p,
            data=utils_transform_provider_data(
                dataset.loc[dataset.index.isin(d),], p, object_type
            ),
            object_type=object_type,
        )
        for p, d in grouped.groups.items()
    ]
    syncables = [k for k in syncables if len(k.data) > 0]
    return syncables


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
        player_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                player_data = player_data[
                    player_data[f"{provider}_{object_type}_id"].isna()
                ]
            else:
                player_data = player_data[
                    player_data[f"{provider}_{object_type}_id"] == provider_id
                ]

        assert len(player_data) == 1


if __name__ == "__main__":
    test_cases = [
        ## player
        # base case
        {
            "file_path": "2025-07-05-usa-mex.csv",
            "object_type": "player",
            "expected_object_ids": [
                {"provider_a": "332705", "provider_b": "12751", "provider_c": "24629"}
            ],
        },
        # same name twice but two different players
        {
            "file_path": "2025-07-15-kor-jpn.csv",
            "object_type": "player",
            "expected_object_ids": [
                {"provider_a": "508366", "provider_b": "95406", "provider_c": "155852"},
                {"provider_a": "645847", "provider_b": "96073", "provider_c": "73729"},
            ],
        },
        # women's game + unused subs + spotty birth dates
        {
            "file_path": "2024-09-22-chi-sd-w.csv",
            "object_type": "player",
            "expected_object_ids": [
                {"provider_a": None, "provider_b": "218500"},  # unused sub, not in A
                {"provider_a": "27889", "provider_b": "33135"},  # naive name matching
                {"provider_a": "13556", "provider_b": "5041"},  # no birth date
            ],
        },
        # disjoint player sets between data providers
        {
            "file_path": "2023-09-13-oma-usa.csv",
            "object_type": "player",
            "expected_object_ids": [
                {"provider_a": None, "provider_b": "190928"},
                {"provider_a": "429448", "provider_b": None},
            ],
        },
        ## match
        # base case + rescheduled
        {
            "file_path": "2025-mls.csv",
            "object_type": "match",
            "expected_object_ids": [
                {
                    "provider_a": "3981151",
                    "provider_b": "4513981",
                    "provider_c": "2004931",
                }
            ],
        },
        # coverage differences
        {
            "file_path": "2024-25-ucl.csv",
            "object_type": "match",
            "expected_object_ids": [{"provider_a": "3945546", "provider_b": None}],
        },
        ## team
        # base case
        {
            "file_path": "2025-nwsl.csv",
            "object_type": "team",
            "expected_object_ids": [
                {"provider_a": "21983", "provider_b": "13449", "provider_c": "3485"}
            ],
        },
        # coverage differences
        {
            "file_path": "2024-25-ucl.csv",
            "object_type": "team",
            "expected_object_ids": [
                {"provider_a": "1028", "provider_b": "197"},
                {"provider_a": "957", "provider_b": None},
            ],
        },
    ]

    for i, c in enumerate(test_cases):
        print(f"\n\n---- TEST CASE {i} - STARTING ------")
        print(f"---- TEST CASE {i} - PARAMS: {c} ------")
        test_synchronize(**c)
        print(f"---- TEST CASE {i} - PASSED ------\n\n")
