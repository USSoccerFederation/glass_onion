"""
Synthetic 3+ provider player synchronization scenarios.

Every scenario is run against every ordering of `content`: the result of a
synchronization should not depend on the order providers are passed in.
"""

from itertools import permutations
from typing import Callable
import pandas as pd
import pytest

from glass_onion.player import PlayerSyncableContent, PlayerSyncEngine

# (player_name, player_nickname, jersey_number, team_id, birth_date)
PLAYERS = [
    ("Christian Pulisic", "Christian Pulisic", "10", "USA", "1998-09-18"),
    ("Weston McKennie", "Weston McKennie", "8", "USA", "1998-08-28"),
    ("Timothy Weah", "Tim Weah", "21", "USA", "2000-02-22"),
    ("Guillermo Ochoa", "Memo Ochoa", "13", "MEX", "1985-07-13"),
    ("Edson Alvarez", "Edson Álvarez", "4", "MEX", "1997-10-24"),
]

# each provider's IDs are offset so they never collide: provider_a -> 100 + i, etc.
ID_BASES = {"provider_a": 100, "provider_b": 200, "provider_c": 300, "provider_d": 400}


def create_provider(
    provider: str,
    overrides: dict[int, dict[str, str]] = {},
) -> PlayerSyncableContent:
    rows = []
    for i, (name, nickname, jersey, team, birth_date) in enumerate(PLAYERS):
        row = {
            f"{provider}_player_id": str(ID_BASES[provider] + i),
            "player_name": name,
            "player_nickname": nickname,
            "jersey_number": jersey,
            "team_id": team,
            "birth_date": birth_date,
        }
        row.update(overrides.get(i, {}))
        rows.append(row)
    return PlayerSyncableContent(provider, pd.DataFrame(rows))


def scenario_independent() -> list[PlayerSyncableContent]:
    return [create_provider(p) for p in ["provider_a", "provider_b", "provider_c"]]


def scenario_four_providers() -> list[PlayerSyncableContent]:
    return [create_provider(p) for p in ID_BASES.keys()]


def scenario_bridged() -> list[PlayerSyncableContent]:
    # provider_a and provider_b can't match player 2 directly (disjoint names,
    # different jerseys), but provider_c matches each of them on its own:
    # its player_name matches provider_a and its player_nickname matches provider_b.
    return [
        create_provider(
            "provider_a",
            overrides={2: {"player_nickname": "Timothy Weah", "jersey_number": "21"}},
        ),
        create_provider(
            "provider_b",
            overrides={
                2: {
                    "player_name": "T. Weah",
                    "player_nickname": "T. Weah",
                    "jersey_number": "22",
                }
            },
        ),
        create_provider(
            "provider_c",
            overrides={2: {"player_nickname": "T. Weah", "jersey_number": "23"}},
        ),
    ]


def ordered(
    scenario: Callable[[], list[PlayerSyncableContent]],
) -> list:
    providers = [c.provider for c in scenario()]
    return [
        pytest.param(
            scenario.__name__,
            list(p),
            id="-".join(x.removeprefix("provider_") for x in p),
        )
        for p in permutations(providers)
    ]


def run_scenario(
    scenario: Callable[[], list[PlayerSyncableContent]], order: list[str]
) -> pd.DataFrame:
    content = {c.provider: c for c in scenario()}
    engine = PlayerSyncEngine([content[p] for p in order], verbose=False)
    return engine.synchronize().data


def assert_fully_unified(result: pd.DataFrame, providers: list[str]):
    id_fields = [f"{p}_player_id" for p in providers]
    for i in range(len(PLAYERS)):
        expected = {f: str(ID_BASES[p] + i) for f, p in zip(id_fields, providers)}
        matches = result
        for field, value in expected.items():
            matches = matches[matches[field] == value]
        assert len(matches) == 1, (
            f"Expecting one row with IDs: {expected}, Actual rows: "
            + result[id_fields].to_json(orient="records", index=False)
        )

    assert len(result) == len(PLAYERS), (
        f"Expecting {len(PLAYERS)} rows, got {len(result)}: "
        + result[id_fields].to_json(orient="records", index=False)
    )


@pytest.mark.parametrize(
    "scenario_name, order",
    ordered(scenario_independent) + ordered(scenario_four_providers),
)
def test_synchronize_all_providers_match(scenario_name: str, order: list[str]):
    result = run_scenario(globals()[scenario_name], order)
    assert_fully_unified(result, sorted(order))


@pytest.mark.parametrize("scenario_name, order", ordered(scenario_bridged))
def test_synchronize_bridged_by_third_provider(scenario_name: str, order: list[str]):
    result = run_scenario(globals()[scenario_name], order)
    assert_fully_unified(result, sorted(order))
