import time
from typing import Callable

from glass_onion.resolver import ObjectResolver, Record, resolve

PROVIDERS = ["provider_a", "provider_b", "provider_c"]

# each provider's IDs are offset so they never collide: provider_a -> 1_000_000 + i, etc.
ID_BASES = {"provider_a": 1_000_000, "provider_b": 2_000_000, "provider_c": 3_000_000}

# a provider_a ID that no player has, used to make a duplicate
DUPLICATE_OFFSET = 500_000

SCALE_FACTOR = 8
# linear work grows ~8x (up to ~14x with overhead); quadratic work grows ~64x
MAX_GROWTH = 32


def record(source: str, i: int, **ids: str) -> Record:
    return {
        "source": source,
        "ids": {f"{p}_player_id": v for p, v in ids.items()},
        "meta": {"player_name": f"Player {i}"},
    }


def player(source: str, i: int) -> Record:
    # player i, synced across every provider
    return record(source, i, **{p: str(ID_BASES[p] + i) for p in PROVIDERS})


def best_time(fn: Callable[[], object], runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def assert_linear(fn: Callable[[int], object], n: int):
    small = best_time(lambda: fn(n))
    large = best_time(lambda: fn(n * SCALE_FACTOR))
    assert large / small < MAX_GROWTH, (
        f"{SCALE_FACTOR}x the input took {large / small:.1f}x as long ({small:.3f}s -> {large:.3f}s)"
    )


def test_resolve_season_of_repeated_syncs():
    # a season of matchdays, each re-syncing every known player and adding a few new ones. Repeats agree with the
    # resolver, so they're accepted, but each player keeps only the record that first linked them.
    players, matchdays, new_per_matchday = 500, 38, 5
    resolver = ObjectResolver()
    for md in range(matchdays):
        roster = players + md * new_per_matchday
        accepted, rejected = resolve(
            resolver, [player(f"matchday_{md}", i) for i in range(roster)]
        )
        assert len(accepted) == roster
        assert rejected == []

    total = players + (matchdays - 1) * new_per_matchday
    components = resolver.components()
    assert len(components) == total
    assert all(len(records) == 1 for _, records in components)

    # a late conflict clashes with the player's one stored record, not one per matchday
    conflicting = record(
        "late",
        0,
        provider_a=str(ID_BASES["provider_a"] + DUPLICATE_OFFSET),
        provider_b=str(ID_BASES["provider_b"]),
    )
    accepted, rejected = resolve(resolver, [conflicting])
    assert accepted == []
    assert rejected == [(conflicting, [player("matchday_0", 0)])]


def test_add_record_merge_cost_is_linear():
    # every record links a new vertex to one ever-growing component. Merging the larger side into the smaller would
    # copy the whole component each time.
    def build(n: int) -> ObjectResolver:
        resolver = ObjectResolver()
        resolver.add_record(record("seed", 0, provider_a="seed", provider_b="0"))
        for i in range(n):
            resolver.add_record(
                {
                    "source": "s",
                    "ids": {f"p{i}_player_id": str(i), "provider_b_player_id": "0"},
                    "meta": {},
                }
            )
        return resolver

    resolver = build(1_000)
    assert len(resolver.components()) == 1
    assert len(resolver.components()[0][0]) == 1_000 + 2

    assert_linear(build, 2_500)


def test_resolve_throughput_is_linear():
    # one batch of new players, each synced across every provider
    def run(n: int) -> ObjectResolver:
        resolver = ObjectResolver()
        accepted, rejected = resolve(
            resolver, [player("matchday_0", i) for i in range(n)]
        )
        assert len(accepted) == n
        assert rejected == []
        return resolver

    assert len(run(1_000).components()) == 1_000

    assert_linear(run, 2_500)


def test_resolve_conflicts_at_scale_are_linear():
    # every 100th player also has a duplicate provider_a ID synced to their other IDs in the same batch. The pair
    # conflict with each other, so both are rejected; every other player is accepted.
    def batch(n: int) -> list[Record]:
        records = [player("matchday_0", i) for i in range(n)]
        for i in range(0, n, 100):
            records.append(
                record(
                    "matchday_0",
                    i,
                    provider_a=str(ID_BASES["provider_a"] + DUPLICATE_OFFSET + i),
                    provider_b=str(ID_BASES["provider_b"] + i),
                    provider_c=str(ID_BASES["provider_c"] + i),
                )
            )
        return records

    def run(n: int) -> tuple[list[Record], list[tuple[Record, list[Record]]]]:
        return resolve(ObjectResolver(), batch(n))

    n = 1_000
    accepted, rejected = run(n)
    duplicated = n // 100
    assert len(accepted) == n - duplicated
    assert len(rejected) == 2 * duplicated
    # each rejected record clashes only with its own pair, not with other conflicts in the batch
    assert all(len(clashes) == 1 for _, clashes in rejected)

    assert_linear(run, 2_500)
