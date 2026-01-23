import pytest
import pandas as pd
import re
from pandera.errors import SchemaError
from itertools import permutations

from glass_onion.match import MatchSyncEngine, MatchSyncableContent


def test_init_syncable_content_null_competition_id():
    with pytest.raises(
        SchemaError,
        match=re.escape("non-nullable series 'competition_id' contains null values"),
    ):
        MatchSyncableContent(
            "provider_a",
            pd.DataFrame(
                [
                    {
                        "provider_a_match_id": "1",
                        "match_date": "2026-01-01",
                        "home_team_id": "1",
                        "away_team_id": "2",
                        "competition_id": pd.NA,
                        "season_id": "1",
                    }
                ]
            ),
        )


def test_init_competition_context():
    engine = MatchSyncEngine(
        content=[
            MatchSyncableContent(
                "provider_a",
                pd.DataFrame(
                    columns=[
                        "provider_a_match_id",
                        "match_date",
                        "home_team_id",
                        "away_team_id",
                        "competition_id",
                        "season_id",
                    ]
                ),
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


def test_init_competition_context_missing_competition_id():
    content_a = MatchSyncableContent(
        "provider_a",
        pd.DataFrame(
            columns=[
                "provider_a_match_id",
                "match_date",
                "home_team_id",
                "away_team_id",
                "season_id",
            ]
        ),
    )

    with pytest.raises(
        SchemaError,
        match=re.escape(
            "column 'competition_id' not in dataframe. Columns in dataframe: ['provider_a_match_id', 'match_date', 'home_team_id', 'away_team_id', 'season_id']"
        ),
    ):
        MatchSyncEngine(
            content=[content_a],
            use_competition_context=True,
        )


@pytest.mark.parametrize(
    "a_match_date, b_match_date, expose_matchday, n_synchronize_on_adjusted_dates, n_synchronize_on_matchday, expected_matches",
    [
        # # ensure no methods hit with perfect match
        ("2025-01-01", "2025-01-01", False, 0, 0, 1),
        # # ensure only adjusted dates if one day away and no matchday
        ("2025-01-01", "2025-01-02", False, 12, 0, 1),
        # # ensure no matches if not the same date + no matchday
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
                    "provider_a_match_id": "1",
                    "matchday": 1,
                    "match_date": a_match_date,
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    right = MatchSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_match_id": "1",
                    "matchday": 1,
                    "match_date": b_match_date,
                    "home_team_id": "1",
                    "away_team_id": "2",
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
    "middle_match_date, middle_matchday, expected_rows, expected_matches",
    [
        # one day off, same game -- A/B/C should sync
        ("2025-01-02", "1", 1, 1),
        # # one day off, same game, don't use matchday -- A/B/C should sync
        ("2025-01-02", None, 1, 1),
        # clearly a different game -- A and C should sync, B separate
        ("2025-02-01", "2", 2, 1),
        # clearly a different game, no matchday -- A and C should sync, B separate
        ("2025-02-01", None, 2, 1),
    ],
)
def test_synchronize_three_levels(
    middle_match_date: str,
    middle_matchday: str,
    expected_rows: int,
    expected_matches: int,
):
    left = MatchSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_match_id": "1",
                    "matchday": "1",
                    "match_date": "2025-01-01",
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    middle = MatchSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_match_id": "1",
                    "matchday": middle_matchday,
                    "match_date": middle_match_date,
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    right = MatchSyncableContent(
        "provider_c",
        data=pd.DataFrame(
            [
                {
                    "provider_c_match_id": "1",
                    "matchday": "1",
                    "match_date": "2025-01-02",
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    engine = MatchSyncEngine([left, middle, right], verbose=True)

    result = engine.synchronize()

    assert set(
        [
            "match_date",
            "provider",
            "home_team_id",
            "away_team_id",
            "provider_a_match_id",
            "provider_b_match_id",
            "provider_c_match_id",
        ]
    ) == set(result.data.columns)

    assert len(result.data) == expected_rows
    assert (
        len(
            result.data[
                result.data["provider_a_match_id"].notna()
                & result.data["provider_c_match_id"].notna()
            ]
        )
        == expected_matches
    )


def test_synchronize_three_levels_no_B_match_iterations():
    left = MatchSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_match_id": "1",
                    "matchday": "1",
                    "match_date": "2025-01-01",
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    middle = MatchSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_match_id": "1",
                    "matchday": "2",
                    "match_date": "2025-02-01",
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    right = MatchSyncableContent(
        "provider_c",
        data=pd.DataFrame(
            [
                {
                    "provider_c_match_id": "1",
                    "matchday": "1",
                    "match_date": "2025-01-02",
                    "home_team_id": "1",
                    "away_team_id": "2",
                }
            ]
        ),
    )

    options = permutations([left, middle, right], 3)
    for p in options:
        content = list(p)
        id_mask = list([c.id_field for c in content])
        engine = MatchSyncEngine(content, verbose=True)
        result = engine.synchronize()

        assert set(
            [
                "match_date",
                "provider",
                "home_team_id",
                "away_team_id",
                "provider_a_match_id",
                "provider_b_match_id",
                "provider_c_match_id",
            ]
        ) == set(result.data.columns), (
            f"Expected columns did not match actual columns for iteration ({id_mask})"
        )

        assert len(result.data) == 2, (
            f"Expected rows did not match actual rows for iteration ({id_mask})"
        )
        assert (
            len(
                result.data[
                    result.data["provider_a_match_id"].notna()
                    & result.data["provider_c_match_id"].notna()
                ]
            )
            == 1
        ), (
            f"Expected A & C matches did not match actual A & C matches for iteration ({id_mask})"
        )
