"""
Synthetic 3-provider scenarios that synchronize players with PlayerSyncEngine, then resolve the synchronized rows into an ObjectResolver.

Every scenario is run against every ordering of providers: neither synchronization nor resolution should depend on the order providers are passed in.
"""

from itertools import permutations
import pandas as pd
import pytest

from glass_onion.player import PlayerSyncableContent, PlayerSyncEngine
from glass_onion.resolver import ObjectResolver, Record

# (player_name, player_nickname, jersey_number, team_id, birth_date)
PLAYERS = [
    ("Christian Pulisic", "Christian Pulisic", "10", "USA", "1998-09-18"),
    ("Weston McKennie", "Weston McKennie", "8", "USA", "1998-08-28"),
    ("Timothy Weah", "Tim Weah", "21", "USA", "2000-02-22"),
    ("Guillermo Ochoa", "Memo Ochoa", "13", "MEX", "1985-07-13"),
    ("Edson Alvarez", "Edson Álvarez", "4", "MEX", "1997-10-24"),
    ("Tyler Adams", "Tyler Adams", "4", "USA", "1999-02-14"),
    ("Hirving Lozano", "Chucky Lozano", "22", "MEX", "1995-07-30"),
    ("Folarin Balogun", "Folarin Balogun", "20", "USA", "2001-07-03"),
]

# players 0-4 are listed by every provider; 5-7 are each listed by only one
SHARED = [0, 1, 2, 3, 4]
ONLY_IN = {"provider_a": 5, "provider_b": 6, "provider_c": 7}

PROVIDERS = ["provider_a", "provider_b", "provider_c"]

# each provider's IDs are offset so they never collide: provider_a -> 100 + i, etc.
ID_BASES = {"provider_a": 100, "provider_b": 200, "provider_c": 300}

# a provider's duplicate ID for player i: provider_a -> 150 + i, etc.
DUPLICATE_OFFSET = 50


def player_id(provider: str, i: int) -> str:
    return str(ID_BASES[provider] + i)


def duplicate_id(provider: str, i: int) -> str:
    return str(ID_BASES[provider] + DUPLICATE_OFFSET + i)


def create_provider(
    provider: str,
    players: list[int],
    ids: dict[int, str] = {},
    extra: list[tuple[int, str, dict[str, str]]] = [],
) -> PlayerSyncableContent:
    """
    Builds a provider's player data. `ids` replaces the default ID for a player, and `extra` adds `(player, ID, overrides)` rows, e.g. duplicates.
    """
    rows = []
    for i, pid, overrides in [
        (i, ids.get(i, player_id(provider, i)), {}) for i in players
    ] + extra:
        name, nickname, jersey, team, birth_date = PLAYERS[i]
        row = {
            f"{provider}_player_id": pid,
            "player_name": name,
            "player_nickname": nickname,
            "jersey_number": jersey,
            "team_id": team,
            "birth_date": birth_date,
        }
        row.update(overrides)
        rows.append(row)
    return PlayerSyncableContent(provider, pd.DataFrame(rows))


def synchronize(content: list[PlayerSyncableContent]) -> pd.DataFrame:
    return PlayerSyncEngine(content, verbose=False).synchronize().data


def to_records(synced: pd.DataFrame, source: str) -> list[Record]:
    # missing IDs are passed through as-is: the resolver is responsible for ignoring them
    id_fields = [c for c in synced.columns if c.endswith("_player_id")]
    return [
        {
            "source": source,
            "ids": {f: row[f] for f in id_fields},
            "meta": {"player_name": row["player_name"]},
        }
        for _, row in synced.iterrows()
    ]


def sync_and_resolve(
    resolver: ObjectResolver,
    content: dict[str, PlayerSyncableContent],
    order: tuple[str, ...],
    source: str,
) -> tuple[list[Record], list[tuple[Record, list[Record]]]]:
    return resolver.resolve(
        to_records(synchronize([content[p] for p in order]), source)
    )


def ids_of(*pairs: tuple[str, int | str]) -> frozenset:
    # e.g. ids_of(("provider_a", 0), ("provider_b", "250")): an int is a player index, a str is a literal ID
    return frozenset(
        (f"{p}_player_id", player_id(p, i) if isinstance(i, int) else i)
        for p, i in pairs
    )


def every_provider(i: int) -> frozenset:
    return ids_of(*[(p, i) for p in PROVIDERS])


def objects(resolver: ObjectResolver) -> set[frozenset]:
    """
    Each object's IDs. Built from the resolver's vertices instead of `components()`, whose `{provider: ID}` dicts would hide two IDs from one provider.
    """
    by_root: dict = {}
    for v in resolver.parent:
        by_root.setdefault(resolver.find(v), set()).add(v)
    return {frozenset(vs) for vs in by_root.values()}


def assert_one_id_per_provider(resolver: ObjectResolver):
    for ids in objects(resolver):
        fields = [f for f, _ in ids]
        assert len(fields) == len(set(fields)), (
            f"Object has two IDs from one provider: {sorted(ids)}"
        )


def record_ids(record: Record) -> frozenset:
    return frozenset((f, i) for f, i in record["ids"].items() if pd.notna(i))


ORDERS = [
    pytest.param(p, id="-".join(x.removeprefix("provider_") for x in p))
    for p in permutations(PROVIDERS)
]


@pytest.mark.parametrize("order", ORDERS)
def test_resolve_players_in_every_provider(order: tuple[str, ...]):
    content = {p: create_provider(p, SHARED) for p in PROVIDERS}

    resolver = ObjectResolver()
    accepted, rejected = sync_and_resolve(resolver, content, order, "matchday_1")

    assert len(accepted) == len(SHARED)
    assert rejected == []
    assert objects(resolver) == {every_provider(i) for i in SHARED}


@pytest.mark.parametrize("order", ORDERS)
def test_resolve_players_in_one_provider(order: tuple[str, ...]):
    # each provider lists one player the others don't: those stay separate, single-ID objects
    content = {p: create_provider(p, SHARED + [ONLY_IN[p]]) for p in PROVIDERS}

    resolver = ObjectResolver()
    accepted, rejected = sync_and_resolve(resolver, content, order, "matchday_1")

    assert len(accepted) == len(SHARED) + len(ONLY_IN)
    assert rejected == []
    assert objects(resolver) == {every_provider(i) for i in SHARED} | {
        ids_of((p, i)) for p, i in ONLY_IN.items()
    }
    # the single-ID rows' missing IDs didn't link them to each other
    assert_one_id_per_provider(resolver)


@pytest.mark.parametrize("duplicated", PROVIDERS)
@pytest.mark.parametrize("order", ORDERS)
def test_resolve_duplicate_player_in_one_provider(
    order: tuple[str, ...], duplicated: str
):
    # one provider lists Christian Pulisic twice, the second time under another ID and a variant name. The duplicate can't
    # join the synced player (that would give it two IDs from one provider), so it becomes its own single-ID object.
    content = {p: create_provider(p, SHARED) for p in PROVIDERS}
    content[duplicated] = create_provider(
        duplicated,
        SHARED,
        extra=[
            (
                0,
                duplicate_id(duplicated, 0),
                {"player_name": "C. Pulisic", "player_nickname": "Pulisic"},
            )
        ],
    )

    resolver = ObjectResolver()
    accepted, rejected = sync_and_resolve(resolver, content, order, "matchday_1")

    assert len(accepted) == len(SHARED) + 1
    assert rejected == []
    assert objects(resolver) == {every_provider(i) for i in SHARED} | {
        ids_of((duplicated, duplicate_id(duplicated, 0)))
    }
    assert_one_id_per_provider(resolver)


@pytest.mark.parametrize("first", ["original", "duplicate"])
@pytest.mark.parametrize("duplicated", PROVIDERS)
def test_resolve_exact_duplicate_player_in_one_provider(duplicated: str, first: str):
    # as above, but the duplicate row matches the original on every column except its ID. synchronize() deduplicates on
    # join_columns and keeps the first ID listed, so only that one reaches the resolver.
    original = (0, player_id(duplicated, 0), {})
    duplicate = (0, duplicate_id(duplicated, 0), {})
    rest = [(i, player_id(duplicated, i), {}) for i in SHARED[1:]]
    rows = [original, duplicate] if first == "original" else [duplicate, original]
    content = {p: create_provider(p, SHARED) for p in PROVIDERS}
    content[duplicated] = create_provider(duplicated, [], extra=rows + rest)
    kept, dropped = rows[0][1], rows[1][1]

    resolver = ObjectResolver()
    accepted, rejected = sync_and_resolve(
        resolver, content, tuple(PROVIDERS), "matchday_1"
    )

    assert len(accepted) == len(SHARED)
    assert rejected == []
    pulisic = (every_provider(0) - ids_of((duplicated, 0))) | ids_of((duplicated, kept))
    assert objects(resolver) == {every_provider(i) for i in SHARED[1:]} | {pulisic}
    assert (f"{duplicated}_player_id", dropped) not in resolver


@pytest.mark.parametrize("order", ORDERS)
def test_resolve_later_sync_links_single_provider_player(order: tuple[str, ...]):
    # Tyler Adams is only in provider_a on matchday 1; on matchday 2 provider_b lists him too, and the new link joins the
    # existing object. Folarin Balogun appears for the first time, in provider_c only.
    resolver = ObjectResolver()
    matchday_1 = {p: create_provider(p, SHARED) for p in PROVIDERS}
    matchday_1["provider_a"] = create_provider("provider_a", SHARED + [5])
    sync_and_resolve(resolver, matchday_1, order, "matchday_1")
    assert ids_of(("provider_a", 5)) in objects(resolver)

    matchday_2 = {
        "provider_a": create_provider("provider_a", SHARED + [5]),
        "provider_b": create_provider("provider_b", SHARED + [5]),
        "provider_c": create_provider("provider_c", SHARED + [7]),
    }
    accepted, rejected = sync_and_resolve(resolver, matchday_2, order, "matchday_2")

    assert len(accepted) == len(SHARED) + 2
    assert rejected == []
    assert objects(resolver) == {every_provider(i) for i in SHARED} | {
        ids_of(("provider_a", 5), ("provider_b", 5)),
        ids_of(("provider_c", 7)),
    }


@pytest.mark.parametrize("duplicated", PROVIDERS)
@pytest.mark.parametrize("order", ORDERS)
def test_resolve_later_sync_rejects_duplicate_id(
    order: tuple[str, ...], duplicated: str
):
    # on matchday 2, one provider lists Christian Pulisic under a new, duplicate ID. The other two providers still sync him
    # to it, but the resolver already holds his original ID from that provider, so the synced row is rejected.
    resolver = ObjectResolver()
    matchday_1 = {p: create_provider(p, SHARED) for p in PROVIDERS}
    sync_and_resolve(resolver, matchday_1, order, "matchday_1")
    original = next(
        r
        for _, rs in resolver.components()
        for r in rs
        if record_ids(r) == every_provider(0)
    )

    matchday_2 = {p: create_provider(p, SHARED) for p in PROVIDERS}
    matchday_2[duplicated] = create_provider(
        duplicated, SHARED, ids={0: duplicate_id(duplicated, 0)}
    )
    accepted, rejected = sync_and_resolve(resolver, matchday_2, order, "matchday_2")

    duplicate_row = (every_provider(0) - ids_of((duplicated, 0))) | ids_of(
        (duplicated, duplicate_id(duplicated, 0))
    )
    assert [record_ids(r) for r, _ in rejected] == [duplicate_row]
    assert rejected[0][1] == [original]
    assert sorted(map(record_ids, accepted), key=sorted) == sorted(
        (every_provider(i) for i in SHARED[1:]), key=sorted
    )
    # the duplicate ID never entered the resolver; the original object is unchanged
    assert (f"{duplicated}_player_id", duplicate_id(duplicated, 0)) not in resolver
    assert objects(resolver) == {every_provider(i) for i in SHARED}


@pytest.mark.parametrize("order", ORDERS)
def test_resolve_later_sync_mixed_batch(order: tuple[str, ...]):
    # matchday 2 combines every case: a duplicate ID for a synced player (rejected), a single-provider player gaining a
    # second provider (accepted), a player new to one provider (accepted), and unchanged players (accepted)
    matchday_1 = {p: create_provider(p, SHARED) for p in PROVIDERS}
    matchday_1["provider_a"] = create_provider("provider_a", SHARED + [5])
    truth = to_records(synchronize([matchday_1[p] for p in order]), "matchday_1")

    matchday_2 = {
        "provider_a": create_provider("provider_a", SHARED + [5]),
        "provider_b": create_provider(
            "provider_b", SHARED + [5], ids={2: duplicate_id("provider_b", 2)}
        ),
        "provider_c": create_provider("provider_c", SHARED + [6]),
    }
    records = to_records(synchronize([matchday_2[p] for p in order]), "matchday_2")

    # resolving the same rows in reverse gives the same result
    results = []
    for batch in (records, records[::-1]):
        r = ObjectResolver()
        r.resolve(truth)
        accepted, rejected = r.resolve(batch)
        results.append(
            (
                {record_ids(x) for x in accepted},
                {record_ids(x) for x, _ in rejected},
                objects(r),
            )
        )
    assert results[0] == results[1]

    accepted, rejected, resolved = results[0]
    assert rejected == {
        ids_of(
            ("provider_a", 2),
            ("provider_b", duplicate_id("provider_b", 2)),
            ("provider_c", 2),
        )
    }
    assert accepted == {every_provider(i) for i in [0, 1, 3, 4]} | {
        ids_of(("provider_a", 5), ("provider_b", 5)),
        ids_of(("provider_c", 6)),
    }
    assert resolved == {every_provider(i) for i in SHARED} | {
        ids_of(("provider_a", 5), ("provider_b", 5)),
        ids_of(("provider_c", 6)),
    }
