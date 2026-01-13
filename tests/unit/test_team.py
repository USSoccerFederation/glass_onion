import pytest
from glass_onion.team import TeamSyncEngine, TeamSyncableContent
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
