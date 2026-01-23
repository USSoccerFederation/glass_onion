import pytest
import pandas as pd
import re
from pandera.errors import SchemaError

from glass_onion.team import TeamSyncEngine, TeamSyncableContent

@pytest.mark.parametrize(
    "column",
    [
        "provider_a_team_id",
        "team_name",
        "competition_id",
        "season_id"
    ],
)
def test_init_syncable_content_prevent_mixed_values(column: str):
    base = {
        "provider_a_team_id": "1",
        "team_name": "test",
        "competition_id": "test1",
        "season_id": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_team_id"] = str(i)

        if i % 2 == 1:
            c[column] = pd.NA
        
        dataset.append(c)
    
    df = pd.DataFrame(dataset)

    with pytest.raises(
        SchemaError,
        match=re.escape(f"non-nullable series '{column}' contains null values"),
    ):
        TeamSyncableContent(
            "provider_a",
            df,
        )

def test_init_syncable_content_null_competition_id():
    with pytest.raises(
        SchemaError,
        match=re.escape("non-nullable series 'competition_id' contains null values"),
    ):
        TeamSyncableContent(
            "provider_a",
            pd.DataFrame(
                [
                    {
                        "provider_a_team_id": "1",
                        "team_name": "test",
                        "competition_id": pd.NA,
                        "season_id": "1",
                    }
                ]
            ),
        )


def test_init_competition_context():
    engine = TeamSyncEngine(
        content=[
            TeamSyncableContent(
                "provider_a",
                pd.DataFrame(
                    columns=[
                        "provider_a_team_id",
                        "team_name",
                        "competition_id",
                        "season_id",
                    ]
                ),
            )
        ],
        use_competition_context=True,
    )

    assert engine.join_columns == ["team_name", "competition_id", "season_id"]


def test_init_competition_context_missing_competition_id():
    content_a = TeamSyncableContent(
        "provider_a",
        pd.DataFrame(columns=["provider_a_team_id", "team_name", "season_id"]),
    )

    with pytest.raises(
        SchemaError,
        match=re.escape(
            "column 'competition_id' not in dataframe. Columns in dataframe: ['provider_a_team_id', 'team_name', 'season_id']"
        ),
    ):
        TeamSyncEngine(
            content=[content_a],
            use_competition_context=True,
        )


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
                    "provider_a_team_id": "1",
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
                    "provider_b_team_id": "1",
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
