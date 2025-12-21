import pytest
from glass_onion.match import MatchSyncEngine, MatchSyncableContent
from tests.utils import utils_create_syncables, FIXTURE_DATA_PATH
import pandas as pd


def test_init_competition_context():
    engine = MatchSyncEngine(
        content=[
            MatchSyncableContent(
                "provider_a", pd.DataFrame(columns=["provider_a_match_id"])
            )
        ],
        use_competition_context=True,
    )

    assert engine.join_columns == [
        "match_date",
        "competition_id",
        "season_id",
        "home_team_id",
        "away_team_id",
    ]


@pytest.mark.parametrize(
    "a_match_date, b_match_date, expose_matchday, n_synchronize_on_adjusted_dates, n_synchronize_on_matchday, expected_matches",
    [
        # ensure no methods hit with perfect match
        ("2025-01-01", "2025-01-01", False, 0, 0, 1),
        # ensure only adjusted dates if one day away and no matchday
        ("2025-01-01", "2025-01-02", False, 12, 0, 1),
        # ensure no matches if not the same date + no matchday
        ("2025-01-01", "2025-01-08", False, 12, 0, 0),
        # ensure match if not the same date + exposed matchday
        ("2025-01-01", "2025-01-08", True, 12, 1, 1),
    ],
)
def test_synchronize_pair(
    a_match_date: str,
    b_match_date: str,
    expose_matchday: bool,
    n_synchronize_on_adjusted_dates: int,
    n_synchronize_on_matchday: int,
    expected_matches: int,
    mocker,
):
    left = MatchSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_match_id": 1,
                    "matchday": 1,
                    "match_date": a_match_date,
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            ]
        ),
    )

    right = MatchSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_match_id": 1,
                    "matchday": 1,
                    "match_date": b_match_date,
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            ]
        ),
    )

    if not expose_matchday:
        left.data.drop("matchday", axis=1, inplace=True)
        right.data.drop("matchday", axis=1, inplace=True)

    engine = MatchSyncEngine([left, right], verbose=True)
    spy_synchronize_on_adjusted_dates = mocker.spy(
        engine, "synchronize_on_adjusted_dates"
    )
    spy_synchronize_on_matchday = mocker.spy(engine, "synchronize_on_matchday")
    result = engine.synchronize_pair(left, right)
    assert (
        spy_synchronize_on_adjusted_dates.call_count == n_synchronize_on_adjusted_dates
    )
    assert spy_synchronize_on_matchday.call_count == n_synchronize_on_matchday
    if not expose_matchday:
        assert set(
            [
                "match_date",
                "home_team_id",
                "away_team_id",
                "provider_a_match_id",
                "provider_b_match_id",
            ]
        ) == set(result.data.columns)
    else:
        assert set(
            [
                "match_date",
                "home_team_id",
                "away_team_id",
                "provider_a_match_id",
                "provider_b_match_id",
                "matchday",
            ]
        ) == set(result.data.columns)
    assert (
        len(
            result.data[
                (result.data["provider_a_match_id"].notna())
                & (result.data["provider_b_match_id"].notna())
            ]
        )
        == expected_matches
    )


@pytest.mark.parametrize(
    "file_path, expected_object_ids",
    [
        # base case + rescheduled
        (
            "2025-mls.csv",
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
            [{"provider_a": "3945546", "provider_b": None}],
        ),
    ],
)
def test_synchronize_complex_cases(file_path: str, expected_object_ids: dict[str, str]):
    dataset = pd.read_csv(FIXTURE_DATA_PATH / "match" / file_path)

    syncables = utils_create_syncables(dataset, "match")
    engine = MatchSyncEngine(syncables, use_competition_context=False, verbose=False)

    result = engine.synchronize()

    # check different ID conditions/expectations
    for expected_ids in expected_object_ids:
        match_data = result.data
        for provider, provider_id in expected_ids.items():
            if provider_id is None:
                match_data = match_data[match_data[f"{provider}_match_id"].isna()]
            else:
                match_data = match_data[
                    match_data[f"{provider}_match_id"] == provider_id
                ]

        assert len(match_data) == 1
