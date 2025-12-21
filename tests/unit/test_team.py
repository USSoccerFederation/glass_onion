import pytest
from glass_onion.team import TeamSyncEngine, TeamSyncableContent
from tests.utils import utils_create_syncables, FIXTURE_DATA_PATH
import pandas as pd


def test_init_competition_context():
    engine = TeamSyncEngine(
        content=[
            TeamSyncableContent(
                "provider_a", pd.DataFrame(columns=["provider_a_team_id"])
            )
        ],
        use_competition_context=True,
    )

    assert engine.join_columns == ["team_name", "competition_id", "season_id"]


@pytest.mark.parametrize(
    "a_team_name, b_team_name, tries, expected_matches",
    [
        # base case: ensure string similarity never hits with perfect match
        ("Atlanta Beat", "Atlanta Beat", 0, 1),
        # base case: ensure string similarity hits with close to match
        ("Atlanta Beat", "Atlanta Beat WFC", 1, 1),
        # base case: ensure string similarity hits twice with 0 similarity
        ("Atlanta Beat", "South Georgia Tormenta FC", 2, 1),
    ],
)
def test_synchronize_pair(
    a_team_name: str, b_team_name: str, tries: int, expected_matches: int, mocker
):
    left = TeamSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_team_id": 1,
                    "team_name": a_team_name,
                }
            ]
        ),
    )

    right = TeamSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_team_id": 1,
                    "team_name": b_team_name,
                }
            ]
        ),
    )

    engine = TeamSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, "synchronize_with_cosine_similarity")
    result = engine.synchronize_pair(left, right)
    assert spy.call_count == tries
    assert set(["team_name", "provider_a_team_id", "provider_b_team_id"]) == set(
        result.data.columns
    )
    assert len(result.data) == expected_matches


@pytest.mark.parametrize(
    "file_path, expected_object_ids",
    [
        # base case
        (
            "2025-nwsl.csv",
            [{"provider_a": "21983", "provider_b": "13449", "provider_c": "3485"}],
        ),
        # coverage differences
        (
            "2024-25-ucl.csv",
            [
                {"provider_a": "1028", "provider_b": "197"},
                {"provider_a": "957", "provider_b": None},
            ],
        ),
    ],
)
def test_synchronize_complex_cases(file_path: str, expected_object_ids: dict[str, str]):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / "team" / file_path)

    syncables = utils_create_syncables(dataset, "team")
    engine = TeamSyncEngine(syncables, use_competition_context=False, verbose=False)

    result = engine.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_object_ids:
        team_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                team_data = team_data[team_data[f"{provider}_team_id"].isna()]
            else:
                team_data = team_data[team_data[f"{provider}_team_id"] == provider_id]

        assert len(team_data) == 1
